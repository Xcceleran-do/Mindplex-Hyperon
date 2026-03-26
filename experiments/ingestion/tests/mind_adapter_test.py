import unittest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
from collections import defaultdict
from datetime import datetime, timezone

from experiments.ingestion.mind_adapter import (
    _engagement_bucket,
    _normalize_category,
    parse_impression_token,
    _safe_quote,
    _tokenize,
    _length_bucket,
    _reading_time_bucket,
    _tone_bucket,
    _expertise_bucket,
    _primary_goal_bucket,
    _content_type_bucket,
    _sentiment_bucket,
    _date_period_bucket,
    _midpoint_datetime,
    _emit_fact,
    _read_news,
    _update_behavior_stats,
    _iter_dataset_files,
    convert_mind_to_metta,
)


class TestImpressionTokenParsing(unittest.TestCase):
    """Test impression token parsing functionality"""

    def test_parse_impression_token_with_label(self):
        """Test parsing token with impression label"""
        self.assertEqual(parse_impression_token("N123-1"), ("N123", 1))
        self.assertEqual(parse_impression_token("N124-0"), ("N124", 0))

    def test_parse_impression_token_without_label(self):
        """Test parsing token without impression label"""
        self.assertEqual(parse_impression_token("N125"), ("N125", None))

    def test_parse_impression_token_edge_cases(self):
        """Test edge cases in token parsing"""
        self.assertEqual(parse_impression_token(""), (None, None))
        self.assertEqual(parse_impression_token(None), (None, None))
        # Token with hyphen but invalid label is split on last hyphen
        self.assertEqual(parse_impression_token("INVALID-LABEL"), ("INVALID", None))

    def test_parse_impression_token_complex_ids(self):
        """Test parsing tokens with complex news IDs"""
        self.assertEqual(parse_impression_token("N_1234_ABC-1"), ("N_1234_ABC", 1))
        self.assertEqual(parse_impression_token("MN-123-0"), ("MN-123", 0))


class TestEngagementBucket(unittest.TestCase):
    """Test engagement bucketing functionality"""

    def test_engagement_bucket_low(self):
        """Test low engagement classification"""
        self.assertEqual(_engagement_bucket(0.0), "Low")
        self.assertEqual(_engagement_bucket(0.04), "Low")

    def test_engagement_bucket_medium(self):
        """Test medium engagement classification"""
        self.assertEqual(_engagement_bucket(0.05), "Medium")
        self.assertEqual(_engagement_bucket(0.07), "Medium")
        self.assertEqual(_engagement_bucket(0.15), "Medium")

    def test_engagement_bucket_high(self):
        """Test high engagement classification"""
        self.assertEqual(_engagement_bucket(0.16), "High")
        self.assertEqual(_engagement_bucket(0.25), "High")
        self.assertEqual(_engagement_bucket(1.0), "High")

    def test_engagement_bucket_boundaries(self):
        """Test exact boundary values"""
        self.assertEqual(_engagement_bucket(0.049), "Low")
        self.assertEqual(_engagement_bucket(0.05), "Medium")
        self.assertEqual(_engagement_bucket(0.15), "Medium")
        self.assertEqual(_engagement_bucket(0.151), "High")


class TestNormalizeCategory(unittest.TestCase):
    """Test category normalization"""

    def test_normalize_category_basic(self):
        """Test basic category normalization"""
        self.assertEqual(_normalize_category("Sci Tech"), "sci-tech")
        self.assertEqual(_normalize_category("U.S. News"), "us-news")

    def test_normalize_category_whitespace(self):
        """Test category normalization with whitespace"""
        self.assertEqual(_normalize_category("  U.S. News  "), "us-news")
        self.assertEqual(_normalize_category("    Mixed    Case    "), "mixed-case")

    def test_normalize_category_special_chars(self):
        """Test category with special characters removed"""
        # & becomes hyphen from space replacement, then special chars removed
        result1 = _normalize_category("Tech & AI")
        self.assertIn("tech", result1)
        self.assertIn("ai", result1)
        # / is removed, spaces become hyphens
        result2 = _normalize_category("News/Updates")
        self.assertIn("news", result2)
        self.assertIn("updates", result2)

    def test_normalize_category_none_and_empty(self):
        """Test edge cases with None and empty"""
        self.assertEqual(_normalize_category(None), "unknown")
        self.assertEqual(_normalize_category(""), "unknown")

    def test_normalize_category_case_insensitive(self):
        """Test category normalization is case insensitive"""
        self.assertEqual(_normalize_category("Technology"), "technology")
        self.assertEqual(_normalize_category("BREAKING NEWS"), "breaking-news")

    def test_normalize_category_numbers(self):
        """Test category with numbers"""
        self.assertEqual(_normalize_category("5G Technology"), "5g-technology")
        self.assertEqual(_normalize_category("Web3 News"), "web3-news")


class TestSafeQuote(unittest.TestCase):
    """Test safe quoting of values for MeTTa"""

    def test_safe_quote_basic(self):
        """Test basic string quoting"""
        self.assertEqual(_safe_quote("Hello"), "Hello")
        # Quotes are escaped, not removed
        result = _safe_quote('Hello "World"')
        self.assertIn('\\"', result)  # Escaped quotes should be present

    def test_safe_quote_escapes_quotes(self):
        """Test that quotes are properly escaped"""
        result = _safe_quote('Document "with quotes"')
        self.assertIn('\\"', result)

    def test_safe_quote_removes_newlines(self):
        """Test that newlines are replaced with spaces"""
        result = _safe_quote("Multi\nline\ntext")
        self.assertNotIn('\n', result)
        self.assertIn('line', result)

    def test_safe_quote_strips_whitespace(self):
        """Test that leading/trailing whitespace is removed"""
        self.assertEqual(_safe_quote("  hello  "), "hello")

    def test_safe_quote_combined(self):
        """Test combined escaping operations"""
        result = _safe_quote('  "Test" with\nnewline  ')
        self.assertNotIn('\n', result)
        self.assertNotIn('  ', result.strip())


class TestTokenize(unittest.TestCase):
    """Test text tokenization"""

    def test_tokenize_basic(self):
        """Test basic tokenization"""
        tokens = _tokenize("Hello World")
        self.assertEqual(tokens, ["hello", "world"])

    def test_tokenize_lowercase(self):
        """Test tokenization converts to lowercase"""
        tokens = _tokenize("HELLO World")
        self.assertEqual(tokens, ["hello", "world"])

    def test_tokenize_punctuation(self):
        """Test tokenization removes punctuation"""
        tokens = _tokenize("Hello, World! How are you?")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertNotIn(",", tokens)

    def test_tokenize_empty(self):
        """Test tokenization of empty string"""
        self.assertEqual(_tokenize(""), [])
        self.assertEqual(_tokenize(None), [])

    def test_tokenize_preserves_apostrophes(self):
        """Test that apostrophes in contractions are preserved"""
        tokens = _tokenize("don't can't won't")
        self.assertIn("don't", tokens)
        self.assertIn("can't", tokens)


class TestLengthBucket(unittest.TestCase):
    """Test document length bucketing"""

    def test_length_bucket_short(self):
        """Test short document classification"""
        self.assertEqual(_length_bucket(10), "Short")
        self.assertEqual(_length_bucket(17), "Short")

    def test_length_bucket_medium(self):
        """Test medium document classification"""
        self.assertEqual(_length_bucket(18), "Medium")
        self.assertEqual(_length_bucket(30), "Medium")
        self.assertEqual(_length_bucket(44), "Medium")

    def test_length_bucket_long(self):
        """Test long document classification"""
        self.assertEqual(_length_bucket(45), "Long")
        self.assertEqual(_length_bucket(100), "Long")

    def test_length_bucket_boundaries(self):
        """Test exact boundary values"""
        self.assertEqual(_length_bucket(17), "Short")
        self.assertEqual(_length_bucket(18), "Medium")
        self.assertEqual(_length_bucket(44), "Medium")
        self.assertEqual(_length_bucket(45), "Long")


class TestReadingTimeBucket(unittest.TestCase):
    """Test reading time bucketing"""

    def test_reading_time_bucket_short(self):
        """Test short reading time"""
        self.assertEqual(_reading_time_bucket(100, wpm=220), "Short")

    def test_reading_time_bucket_medium(self):
        """Test medium reading time"""
        self.assertEqual(_reading_time_bucket(550, wpm=220), "Medium")

    def test_reading_time_bucket_long(self):
        """Test long reading time"""
        self.assertEqual(_reading_time_bucket(800, wpm=220), "Long")

    def test_reading_time_bucket_custom_wpm(self):
        """Test reading time with custom words per minute"""
        # 200 words at 200 wpm = 1 minute, which is Short
        self.assertEqual(_reading_time_bucket(200, wpm=200), "Short")


class TestToneBucket(unittest.TestCase):
    """Test tone classification"""

    def test_tone_bucket_instructional(self):
        """Test instructional tone detection"""
        self.assertEqual(_tone_bucket("How to Build AI", "Guide to ML"), "Instructional")
        self.assertEqual(_tone_bucket("Tips for Success", "Learn Why AI"), "Instructional")

    def test_tone_bucket_casual(self):
        """Test casual tone detection"""
        self.assertEqual(_tone_bucket("Exclusive Update", "You won't believe"), "Casual")
        self.assertEqual(_tone_bucket("Shocking News", "Amazing AI"), "Casual")

    def test_tone_bucket_formal(self):
        """Test formal tone detection"""
        self.assertEqual(_tone_bucket("Market Analysis", "Economic Report"), "Formal")


class TestExpertiseBucket(unittest.TestCase):
    """Test audience expertise classification"""

    def test_expertise_bucket_beginner(self):
        """Test beginner expertise level"""
        self.assertEqual(_expertise_bucket("Simple AI", "easy guide"), "Beginner")

    def test_expertise_bucket_intermediate(self):
        """Test intermediate expertise level"""
        # "Understanding Deep Learning Neural Networks" has long words
        # This should be Advanced due to high ratio of long words (>0.30)
        result = _expertise_bucket("Understanding Deep Learning", "Neural Networks")
        self.assertIn(result, ["Intermediate", "Advanced"])

    def test_expertise_bucket_advanced(self):
        """Test advanced expertise level"""
        long_words = "Transformer Architecture Implementation ComplexAlgorithm " * 5
        result = _expertise_bucket(long_words, "Mathematics")
        self.assertIn(result, ["Intermediate", "Advanced"])

    def test_expertise_bucket_empty(self):
        """Test with empty text"""
        self.assertEqual(_expertise_bucket("", ""), "Beginner")


class TestPrimaryGoalBucket(unittest.TestCase):
    """Test primary goal classification"""

    def test_primary_goal_entertain(self):
        """Test entertain goal detection"""
        self.assertEqual(_primary_goal_bucket("sports"), "Entertain")
        self.assertEqual(_primary_goal_bucket("entertainment"), "Entertain")

    def test_primary_goal_persuade(self):
        """Test persuade goal detection"""
        self.assertEqual(_primary_goal_bucket("opinion"), "Persuade")

    def test_primary_goal_inform(self):
        """Test inform goal detection (default)"""
        self.assertEqual(_primary_goal_bucket("technology"), "Inform")
        self.assertEqual(_primary_goal_bucket("unknown"), "Inform")


class TestContentTypeBucket(unittest.TestCase):
    """Test content type classification"""

    def test_content_type_opinion(self):
        """Test opinion content type"""
        self.assertEqual(_content_type_bucket("opinion"), "Opinion")

    def test_content_type_news(self):
        """Test news content type (default)"""
        self.assertEqual(_content_type_bucket("technology"), "News")
        self.assertEqual(_content_type_bucket(""), "News")


class TestSentimentBucket(unittest.TestCase):
    """Test sentiment classification"""

    def test_sentiment_bucket_positive(self):
        """Test positive sentiment"""
        self.assertEqual(_sentiment_bucket("Great Success", "Best growth story"), "Positive")

    def test_sentiment_bucket_negative(self):
        """Test negative sentiment"""
        self.assertEqual(_sentiment_bucket("Bad News", "Crisis and risk"), "Negative")

    def test_sentiment_bucket_mixed(self):
        """Test mixed sentiment"""
        self.assertEqual(_sentiment_bucket("Pros and Cons", "Benefits and risks"), "Mixed")


class TestDatePeriodBucket(unittest.TestCase):
    """Test date period bucketing"""

    def test_date_period_recent(self):
        """Test recent date classification"""
        recent_date = datetime.now(timezone.utc)
        self.assertEqual(_date_period_bucket(recent_date), "Recent")

    def test_date_period_last_year(self):
        """Test last year classification"""
        from datetime import timedelta
        old_date = datetime.now(timezone.utc) - timedelta(days=200)
        self.assertEqual(_date_period_bucket(old_date), "Last_Year")

    def test_date_period_archived(self):
        """Test archived classification"""
        from datetime import timedelta
        very_old_date = datetime.now(timezone.utc) - timedelta(days=400)
        self.assertEqual(_date_period_bucket(very_old_date), "Archived")

    def test_date_period_none(self):
        """Test with None date"""
        self.assertEqual(_date_period_bucket(None), "Archived")


class TestMidpointDatetime(unittest.TestCase):
    """Test midpoint datetime calculation"""

    def test_midpoint_datetime_single(self):
        """Test midpoint with single value"""
        dt = datetime.now(timezone.utc)
        result = _midpoint_datetime([dt])
        self.assertEqual(result, dt)

    def test_midpoint_datetime_multiple(self):
        """Test midpoint with multiple values"""
        from datetime import timedelta
        dt1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dt2 = datetime(2024, 1, 2, tzinfo=timezone.utc)
        dt3 = datetime(2024, 1, 3, tzinfo=timezone.utc)
        
        result = _midpoint_datetime([dt1, dt3, dt2])
        self.assertEqual(result, dt2)

    def test_midpoint_datetime_empty(self):
        """Test midpoint with empty list"""
        self.assertIsNone(_midpoint_datetime([]))


class TestEmitFact(unittest.TestCase):
    """Test MeTTa fact emission"""

    def test_emit_fact_basic(self):
        """Test basic fact emission"""
        lines = []
        _emit_fact(lines, "prop", "A_1", "value")
        self.assertEqual(len(lines), 1)
        self.assertIn("prop", lines[0])
        self.assertIn("A_1", lines[0])
        self.assertIn("value", lines[0])

    def test_emit_fact_specialchar(self):
        """Test fact emission with special characters"""
        lines = []
        _emit_fact(lines, "prop", "A_1", 'value "with quotes"')
        self.assertEqual(len(lines), 1)

    def test_emit_fact_empty_value(self):
        """Test that empty values are skipped"""
        lines = []
        _emit_fact(lines, "prop", "A_1", "")
        self.assertEqual(len(lines), 0)


class TestConvertMindToMetta(unittest.TestCase):
    """Test MIND dataset conversion (integration tests)"""

    def test_convert_mind_min_documents_validation(self):
        """Test minimum document validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal test data directory
            os.makedirs(os.path.join(temp_dir, "train"))
            
            # Create empty news file
            news_path = os.path.join(temp_dir, "train", "news.tsv")
            with open(news_path, "w") as f:
                f.write("")
            
            with self.assertRaises(RuntimeError) as context:
                convert_mind_to_metta(temp_dir, "output.metta", temp_dir, min_documents=100)
            
            # Check that RuntimeError is raised (message may vary depending on whether records found)
            error_msg = str(context.exception)
            self.assertTrue(
                "No news records" in error_msg or "too few" in error_msg,
                f"Unexpected error message: {error_msg}"
            )

    def test_convert_mind_success(self):
        """Test successful MIND conversion"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create train directory with valid data
            train_dir = os.path.join(temp_dir, "train")
            os.makedirs(train_dir)
            
            # Create news file
            news_path = os.path.join(train_dir, "news.tsv")
            with open(news_path, "w") as f:
                f.write("N1\ttech\tAI\tTitle1\tAbstract1\thttp://example.com\t\t\n")
                f.write("N2\ttech\tML\tTitle2\tAbstract2\thttp://example.com\t\t\n")
            
            output_metta = os.path.join(temp_dir, "output.metta")
            report_dir = os.path.join(temp_dir, "report")
            
            result = convert_mind_to_metta(
                temp_dir, output_metta, report_dir, 
                min_documents=1, max_documents=None
            )
            
            self.assertIn("output_metta_path", result)
            self.assertTrue(os.path.exists(output_metta))


class TestReadNews(unittest.TestCase):
    """Test news file parsing"""

    def test_read_news_valid_file(self):
        """Test reading valid news TSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("N1\ttech\tAI\tTitle\tAbstract\turl\t\t\n")
            f.write("N2\ttech\tML\tTitle2\tAbstract2\turl2\t\t\n")
            f.flush()
            temp_path = f.name
            
        try:
            result = _read_news(temp_path)
            
            self.assertEqual(len(result), 2)
            self.assertIn("N1", result)
            self.assertIn("N2", result)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_read_news_empty_file(self):
        """Test reading empty news file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.flush()
            temp_path = f.name
            
        try:
            result = _read_news(temp_path)
            
            self.assertEqual(len(result), 0)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
