import concurrent.futures
import copy
import logging
from concurrent.futures import as_completed
from typing import List, Union

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import ArticleGenerationModule, Information
from ...utils import ArticleTextProcessing


class StormArticleGenerationModule(ArticleGenerationModule):
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage,
    """

    def __init__(
        self,
        article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retrieve_top_k: int = 5,
        max_thread_num: int = 10,
        target_overall_word_count: int = 500, # Added
        intro_conclusion_word_target: int = 75, # Added
    ):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.target_overall_word_count = target_overall_word_count # Added
        self.intro_conclusion_word_target = intro_conclusion_word_target # Added
        # Pass article_gen_lm to ConvToSection, and later add ShortenSection if implemented
        self.section_gen = ConvToSection(engine=self.article_gen_lm)

    def generate_section(
        self, topic, section_name, information_table, section_outline, section_query, target_word_count: int # Added target_word_count
    ):
        collected_info: List[Information] = []
        if information_table is not None:
            collected_info = information_table.retrieve_information(
                queries=section_query, search_top_k=self.retrieve_top_k
            )
        output = self.section_gen( # This is ConvToSection instance
            topic=topic,
            outline=section_outline, # section_outline is not used by ConvToSection.forward currently
            section=section_name,
            collected_info=collected_info,
            target_word_count=target_word_count, # Pass target_word_count
        )
        return {
            "section_name": section_name,
            "section_content": output.section,
            "collected_info": collected_info,
        }

    def generate_article(
        self,
        topic: str,
        information_table: StormInformationTable,
        article_with_outline: StormArticle,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        """
        Generate article for the topic based on the information table and article outline.

        Args:
            topic (str): The topic of the article.
            information_table (StormInformationTable): The information table containing the collected information.
            article_with_outline (StormArticle): The article with specified outline.
            callback_handler (BaseCallbackHandler): An optional callback handler that can be used to trigger
                custom callbacks at various stages of the article generation process. Defaults to None.
        """
        information_table.prepare_table_for_retrieval()

        if article_with_outline is None:
            logging.warning(f"Article outline for topic '{topic}' is None. Creating a default StormArticle.")
            article_with_outline = StormArticle(topic_name=topic) # Should be topic_name

        sections_to_write = article_with_outline.get_first_level_section_names()

        # Calculate word counts for sections
        num_total_sections = len(sections_to_write)
        main_body_section_titles = []
        has_intro = False
        has_conclusion = False

        for title in sections_to_write:
            if title.lower() == "introduction":
                has_intro = True
            elif title.lower() == "conclusion":
                has_conclusion = True
            else:
                main_body_section_titles.append(title)

        num_main_body_sections = len(main_body_section_titles)

        remaining_wc_for_main_body = float(self.target_overall_word_count)
        if has_intro:
            remaining_wc_for_main_body -= self.intro_conclusion_word_target
        if has_conclusion:
            remaining_wc_for_main_body -= self.intro_conclusion_word_target

        wc_per_main_section = 0
        if num_main_body_sections > 0:
            wc_per_main_section = max(50, int(remaining_wc_for_main_body / num_main_body_sections))
        elif remaining_wc_for_main_body > 0 and num_total_sections > 0 : # Only intro/conclusion
             # if only intro/conclusion, distribute remaining between them if they exist
            if has_intro and has_conclusion:
                self.intro_conclusion_word_target = max(50, int(remaining_wc_for_main_body / (1 if has_intro else 0 + 1 if has_conclusion else 1))))
            elif has_intro or has_conclusion: # only one of them
                 self.intro_conclusion_word_target = max(50, int(remaining_wc_for_main_body))


        section_output_dict_collection = []

        if num_total_sections == 0:
            logging.error(
                f"No outline sections for {topic}. Attempting to write a single section using the topic name."
            )
            section_output_dict = self.generate_section(
                topic=topic,
                section_name=topic,
                information_table=information_table,
                section_outline=topic,
                section_query=[topic],
                target_word_count=self.target_overall_word_count
            )
            section_output_dict_collection = [section_output_dict]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_thread_num
            ) as executor:
                future_to_sec_title = {}
                for section_title in sections_to_write:
                    # We don't want to write a separate introduction section.
                    if section_title.lower().strip() == "introduction":
                        continue
                        # We don't want to write a separate conclusion section.
                    if section_title.lower().strip().startswith(
                        "conclusion"
                    ) or section_title.lower().strip().startswith("summary"):
                        continue
                    section_query = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=False
                    )
                    queries_with_hashtags = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=True
                    )
                    section_outline = "\n".join(queries_with_hashtags)
                    future_to_sec_title[
                        executor.submit(
                            self.generate_section,
                            topic,
                            section_title,
                            information_table,
                            section_outline,
                            section_query,
                        )
                    ] = section_title

                for future in as_completed(future_to_sec_title):
                    section_output_dict_collection.append(future.result())

        article = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            article.update_section(
                parent_section_name=topic,
                current_section_content=section_output_dict["section_content"],
                current_section_info_list=section_output_dict["collected_info"],
            )
        article.post_processing()
        return article


class ConvToSection(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.shorten_section = dspy.Predict(ShortenSection) # Added for length adjustment
        self.engine = engine

    def forward(
        self, topic: str, outline: str, section: str, collected_info: List[Information], target_word_count: int
    ):
        info = ""
        for idx, storm_info in enumerate(collected_info):
            info += f"[{idx + 1}]\n" + "\n".join(storm_info.snippets) # Snippets are already lists of strings
            info += "\n\n"

        info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1500) # Keep limiting input context

        with dspy.settings.context(lm=self.engine):
            generated_section_content = self.write_section(
                topic=topic,
                info=info,
                section=section,
                target_word_count=str(target_word_count) # Ensure it's a string for the prompt
            ).output

            # Optional: Post-generation length adjustment
            current_word_count = len(generated_section_content.split())
            # Allow 30% overshoot before attempting to shorten.
            # Only shorten if current_word_count is meaningfully larger than target.
            if current_word_count > target_word_count * 1.3 and current_word_count > 30: # Avoid shortening very short texts
                logging.info(f"Section '{section}' is too long ({current_word_count} words, target {target_word_count}). Attempting to shorten.")
                generated_section_content = self.shorten_section(
                    section_title=section,
                    target_word_count=str(target_word_count),
                    current_word_count=str(current_word_count),
                    original_text=generated_section_content
                ).shortened_text

        # The new WriteSection prompt asks for content only, no title.
        # So, ArticleTextProcessing.clean_up_section (which removes title) should no longer be needed.
        return dspy.Prediction(section=generated_section_content)


class ShortenSection(dspy.Signature):
    """The following text for the section "{section_title}" is too long. Its target was approximately {target_word_count} words, but it is {current_word_count} words. Please provide a more concise version of this text, strictly adhering to approximately {target_word_count} words. Focus only on the absolute most essential points and retain all citations in their original format (e.g., `[1]`)."""
    section_title = dspy.InputField(prefix="Section title: ")
    target_word_count = dspy.InputField(prefix="Target word count: ")
    current_word_count = dspy.InputField(prefix="Current word count: ")
    original_text = dspy.InputField(prefix="Original text to shorten:\n")
    shortened_text = dspy.OutputField(prefix="Concise version:\n")


class WriteSection(dspy.Signature):
    """You are writing a section titled "{section}" for a brief summary article on the topic "{topic}".
    Your target word count for this specific section is approximately {target_word_count} words.
    Base your writing *only* on the most critical and relevant information from the following knowledge snippets.
    Ensure a neutral, informative, and engaging style suitable for a general audience.
    The content must directly address the section title and flow logically.
    **Crucially, you must cite the source for every piece of factual information using the format `[SNIPPET_ID]`** (e.g., `[1]`, `[2]`).
    Do not include a 'References' or 'Sources' list. Do not use '#' or other hierarchical markers for the section title in your output; write only the paragraph content for this section.

    Provided knowledge snippets:
    {info}

    Write the content for the section "{section}" (approximately {target_word_count} words):
    """

    info = dspy.InputField(prefix="Provided knowledge snippets:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section title to write: ", format=str)
    target_word_count = dspy.InputField(prefix="Target word count for this section: ", format=str)
    output = dspy.OutputField(
        prefix="Write the paragraph content for the section (no title, no hierarchical markers):\n",
        format=str,
    )
