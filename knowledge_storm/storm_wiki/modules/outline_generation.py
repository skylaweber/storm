from typing import Union, Optional, Tuple

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import OutlineGenerationModule
from ...utils import ArticleTextProcessing


class StormOutlineGenerationModule(OutlineGenerationModule):
    """
    The interface for outline generation stage. Given topic, collected information from knowledge
    curation stage, generate outline for the article.
    """

    def __init__(self, outline_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.outline_gen_lm = outline_gen_lm
        self.write_outline = WriteOutline(engine=self.outline_gen_lm)

    def generate_outline(
        self,
        topic: str,
        information_table: StormInformationTable,
        old_outline: Optional[StormArticle] = None,
        callback_handler: BaseCallbackHandler = None,
        return_draft_outline=False,
    ) -> Union[StormArticle, Tuple[StormArticle, StormArticle]]:
        """
        Generates an outline for an article based on the specified topic and the information
        gathered during the knowledge curation stage. This method can optionally return both the
        final article outline and a draft outline if required.

        Args:
            topic (str): The topic of the article.
            information_table (StormInformationTable): The information table containing the collected information.
            old_outline (Optional[StormArticle]): An optional previous version of the article outline that can
                be used for reference or comparison. Defaults to None.
            callback_handler (BaseCallbackHandler): An optional callback handler that can be used to trigger
                custom callbacks at various stages of the outline generation process, such as when the information
                organization starts. Defaults to None.
            return_draft_outline (bool): A flag indicating whether the method should return both the final article
                outline and a draft version of the outline. If False, only the final article outline is returned.
                Defaults to False.

        Returns:
            Union[StormArticle, Tuple[StormArticle, StormArticle]]: Depending on the value of `return_draft_outline`,
                this method returns either a single `StormArticle` object containing the final outline or a tuple of
                two  `StormArticle` objects, the first containing the final outline and the second containing the
                draft outline.
        """
        if callback_handler is not None:
            callback_handler.on_information_organization_start()

        concatenated_dialogue_turns = sum(
            [conv for (_, conv) in information_table.conversations], []
        )
        result = self.write_outline(
            topic=topic,
            dlg_history=concatenated_dialogue_turns,
            callback_handler=callback_handler,
        )

        # result.outline is now expected to be a flat list of section titles, each on a new line.
        section_titles = [title.strip() for title in result.outline.split('\n') if title.strip()]

        article_with_outline_only = StormArticle(topic_name=topic)
        if not section_titles: # Handle empty outline from LLM, though prompt requests sections.
            # Add a default structure if LLM fails to produce one, or log warning.
            # For now, it will be an empty article if section_titles is empty.
            # Consider logging a warning here if section_titles is empty.
            logging.warning(f"Outline generation for topic '{topic}' resulted in an empty list of section titles.")
        for title in section_titles:
            # Add sections directly under the root for a flat structure. Content is empty for now.
            # Corrected class name to ArticleSectionNode from dspy.FixedArticleSectionNode
            new_node = ArticleSectionNode(section_name=title, content="")
            article_with_outline_only.root.add_child(new_node)

        # The result.old_outline contains the original parametrically generated hierarchical outline.
        # This can be used for the "draft_outline" if needed.
        if result.old_outline and result.old_outline.strip():
            article_with_draft_outline_only = StormArticle.from_outline_str(
                topic=topic, outline_str=result.old_outline
            )
        else:
            # If parametric_outline_str was empty for some reason.
            article_with_draft_outline_only = StormArticle(topic_name=topic) # empty article

        if not return_draft_outline:
            return article_with_outline_only
        return article_with_outline_only, article_with_draft_outline_only


class WriteOutline(dspy.Module):
    """Generate the outline for the Wikipedia page."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.draft_page_outline = dspy.Predict(WritePageOutline)
        self.write_page_outline = dspy.Predict(WritePageOutlineFromConv)
        self.engine = engine

    def forward(
        self,
        topic: str,
        dlg_history,
        old_outline: Optional[str] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        trimmed_dlg_history = []
        for turn in dlg_history:
            if (
                "topic you" in turn.agent_utterance.lower()
                or "topic you" in turn.user_utterance.lower()
            ):
                continue
            trimmed_dlg_history.append(turn)
        conv = "\n".join(
            [
                f"Wikipedia Writer: {turn.user_utterance}\nExpert: {turn.agent_utterance}"
                for turn in trimmed_dlg_history
            ]
        )
        conv = ArticleTextProcessing.remove_citations(conv)
        # Reduced word count for curated knowledge to align with short article focus
        conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 1500)

        with dspy.settings.context(lm=self.engine):
            # Generate the parametric outline first (as old_outline was used by the original signature).
            # This might be used as a fallback or ignored by the new prompt in WritePageOutlineFromConv.
            parametric_outline_str = ArticleTextProcessing.clean_up_outline(
                self.draft_page_outline(topic=topic).outline
            )
            if callback_handler:
                callback_handler.on_direct_outline_generation_end(
                    outline=parametric_outline_str
                )

            # The modified WritePageOutlineFromConv now generates the flat list based on `conv`.
            # The old_outline field in its signature is an InputField. We pass parametric_outline_str here.
            # The prompt for WritePageOutlineFromConv instructs the LLM to base its output *only* on conv.
            generated_flat_outline_str = self.write_page_outline(
                topic=topic, old_outline=parametric_outline_str, conv=conv
            ).outline

            # Assuming the LLM follows the new prompt and outputs a clean flat list (each title on a new line).
            # No ArticleTextProcessing.clean_up_outline should be needed for generated_flat_outline_str.
            outline = generated_flat_outline_str

            if callback_handler:
                callback_handler.on_outline_refinement_end(outline=outline)

        # The `old_outline` in the prediction can be the parametric one.
        return dspy.Prediction(outline=outline, old_outline=parametric_outline_str)


class WritePageOutline(dspy.Signature):
    """Write an outline for a Wikipedia page.
    Here is the format of your writing:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Do not include other information.
    3. Do not include topic name itself in the outline.
    """

    topic = dspy.InputField(prefix="The topic you want to write: ", format=str)
    outline = dspy.OutputField(prefix="Write the Wikipedia page outline:\n", format=str)


class NaiveOutlineGen(dspy.Module):
    """Generate the outline with LLM's parametric knowledge directly."""

    def __init__(self):
        super().__init__()
        self.write_outline = dspy.Predict(WritePageOutline)

    def forward(self, topic: str):
        outline = self.write_outline(topic=topic).outline

        return dspy.Prediction(outline=outline)


class WritePageOutlineFromConv(dspy.Signature):
    """You are tasked with creating a simple and logical outline for a short article of about 500 words on the main topic: {topic}.
    Base the outline *only* on the key themes present in the following curated knowledge:
    {conv}

    The outline must follow this structure:
    1. A brief 'Introduction' section.
    2. One or two (at most) main body sections. Each section should cover a distinct key theme found in the curated knowledge. Provide a concise, descriptive title for each main body section.
    3. A brief 'Conclusion' section.

    Output the outline as a flat list of section titles in logical order, with each title on a new line. Do not use '#', '##', or any other hierarchical markers.
    Example output format:
    Introduction
    [Main Theme 1 Title]
    [Main Theme 2 Title (if applicable)]
    Conclusion
    """

    topic = dspy.InputField(prefix="The main topic: ", format=str)
    conv = dspy.InputField(prefix="Curated knowledge (conversation history):\n", format=str)
    # old_outline is no longer strictly needed by the prompt, but kept for signature compatibility if required by dspy.Predict call structure.
    # The prompt instructs the LLM to base the outline *only* on the conversation.
    old_outline = dspy.InputField(prefix="Previous outline (to be ignored by LLM):\n", format=str) # Changed from OutputField to InputField
    outline = dspy.OutputField(
        prefix="Write the flat list of section titles for the short article:\n",
        format=str,
    )
