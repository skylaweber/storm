import copy
from typing import Union

import dspy

from .storm_dataclass import StormArticle
from ...interface import ArticlePolishingModule
from ...utils import ArticleTextProcessing


class StormArticlePolishingModule(ArticlePolishingModule):
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage.
    """

    def __init__(
        self,
        article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm

        self.polish_page = PolishPageModule(
            write_lead_engine=self.article_gen_lm, polish_engine=self.article_polish_lm
        )

    def polish_article(
        self, topic: str, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        """
        Polish article. For short-form articles, this primarily involves a redundancy check
        and ensuring references are correctly processed. Lead section (Introduction) generation
        is now handled in the Article Generation stage.

        Args:
            topic (str): The topic of the article.
            draft_article (StormArticle): The draft article from the generation stage.
            remove_duplicate (bool): Whether to use an LM call to remove duplicates.
        """

        article_text = draft_article.to_string() # This will be the full article string

        # PolishPageModule's forward will now primarily use polish_page if remove_duplicate is True.
        # The lead_section generation within PolishPageModule will be skipped.
        polish_result = self.polish_page( # polish_page is an instance of PolishPageModule
            topic=topic, draft_page=article_text, polish_whole_page=remove_duplicate
        )

        # polish_result.page will be the potentially de-duplicated full article text.
        # polish_result.lead_section will be None or empty as we'll modify PolishPageModule.
        polished_article_text = polish_result.page

        # Reparse the potentially modified text.
        # This assumes PolishPage LLM call correctly maintains the flat structure provided by draft_article.to_string()
        polished_article_dict = ArticleTextProcessing.parse_article_into_dict(
            polished_article_text
        )

        # Create a new StormArticle object to hold the polished version,
        # or update a deepcopy of the draft_article.
        # Using deepcopy ensures that if parsing fails or structure is mangled by LLM,
        # we don't corrupt the original draft_article object before this point.
        polished_article_obj = copy.deepcopy(draft_article)

        # Clear existing content before inserting, to reflect the polished text accurately.
        # A simpler way might be to create a new StormArticle and populate it, if insert_or_create_section
        # doesn't handle removal of sections that might have been deleted by polish_page LLM.
        # However, parse_article_into_dict and insert_or_create_section should rebuild based on parsed dict.
        # For a flat structure, this should be relatively safe.
        # Let's clear the children of root to ensure clean insertion.
        polished_article_obj.root.children = []
        polished_article_obj.insert_or_create_section(article_dict=polished_article_dict)

        # Crucially, transfer the references from the original draft_article,
        # as the LLM polishing calls might not preserve reference metadata, only placeholders.
        polished_article_obj.reference = draft_article.reference

        polished_article_obj.post_processing() # This handles citation reordering etc.
        return polished_article_obj


class WriteLeadSection(dspy.Signature):
    """Write a lead section for the given Wikipedia page with the following guidelines:
    1. The lead should stand on its own as a concise overview of the article's topic. It should identify the topic, establish context, explain why the topic is notable, and summarize the most important points, including any prominent controversies.
    2. The lead section should be concise and contain no more than four well-composed paragraphs.
    3. The lead section should be carefully sourced as appropriate. Add inline citations (e.g., "Washington, D.C., is the capital of the United States.[1][3].") where necessary.
    """

    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    draft_page = dspy.InputField(prefix="The draft page:\n", format=str)
    lead_section = dspy.OutputField(prefix="Write the lead section:\n", format=str)


class PolishPage(dspy.Signature):
    """You are a faithful text editor that is good at finding repeated information in the article and deleting them to make sure there is no repetition in the article. You won't delete any non-repeated part in the article. You will keep the inline citations and article structure (indicated by "#", "##", etc.) appropriately. Do your job for the following ~500-word article."""

    draft_page = dspy.InputField(prefix="The draft ~500-word article:\n", format=str)
    page = dspy.OutputField(prefix="Your revised article (keeping original structure and citations):\n", format=str)


class PolishPageModule(dspy.Module):
    def __init__(
        self,
        write_lead_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        polish_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.write_lead_engine = write_lead_engine
        self.polish_engine = polish_engine
        self.write_lead = dspy.Predict(WriteLeadSection)
        self.polish_page = dspy.Predict(PolishPage)

    def forward(self, topic: str, draft_page: str, polish_whole_page: bool = True):
        # Lead section generation is removed as it's handled by ArticleGeneration stage.
        lead_section_output = None # Or an empty string

        if polish_whole_page:
            # NOTE: Change show_guidelines to false to make the generation more robust to different LM families.
            with dspy.settings.context(lm=self.polish_engine, show_guidelines=False):
                # The PolishPage signature expects draft_page (the full article text)
                page_output = self.polish_page(draft_page=draft_page).page
        else:
            page_output = draft_page

        return dspy.Prediction(lead_section=lead_section_output, page=page_output)
