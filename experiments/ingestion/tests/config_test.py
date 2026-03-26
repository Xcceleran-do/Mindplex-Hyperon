import unittest
from experiments.ingestion.config import (
    LENGTH_BUCKETS, READING_TIME_BUCKETS, ENGAGEMENT_BUCKETS,
    RETENTION_BUCKETS, DETERMINISTIC_STV, AI_FAILURE_STV, UNKNOWN_STV,
    MINDPLEX_API_DOMAIN, USER_ARTICLES_ENDPOINT_TEMPLATE, DEFAULT_USERNAME,
    DEFAULT_HEADERS, ANALYSIS_PROMPT_TEMPLATE
)


class TestConfigConstants(unittest.TestCase):
    """Test suite for configuration constants"""

    def test_length_buckets_structure(self):
        """Test that LENGTH_BUCKETS has correct structure"""
        self.assertIsInstance(LENGTH_BUCKETS, dict)
        self.assertIn("Short", LENGTH_BUCKETS)
        self.assertIn("Medium", LENGTH_BUCKETS)
        self.assertIn("Long", LENGTH_BUCKETS)
        
        # Each bucket should have (condition, stv_range)
        for label, value in LENGTH_BUCKETS.items():
            self.assertIsInstance(value, tuple)
            self.assertEqual(len(value), 2)
            condition, stv_range = value
            self.assertTrue(callable(condition))
            self.assertIsInstance(stv_range, tuple)
            self.assertEqual(len(stv_range), 2)

    def test_length_buckets_conditions(self):
        """Test that LENGTH_BUCKETS conditions work correctly"""
        short_condition, _ = LENGTH_BUCKETS["Short"]
        medium_condition, _ = LENGTH_BUCKETS["Medium"]
        long_condition, _ = LENGTH_BUCKETS["Long"]
        
        # Test boundary conditions
        self.assertTrue(short_condition(250))
        self.assertTrue(medium_condition(1000))
        self.assertTrue(long_condition(2000))

    def test_reading_time_buckets_structure(self):
        """Test that READING_TIME_BUCKETS has correct structure"""
        self.assertIsInstance(READING_TIME_BUCKETS, dict)
        self.assertIn("Very_Short", READING_TIME_BUCKETS)
        self.assertIn("Short", READING_TIME_BUCKETS)
        self.assertIn("Medium", READING_TIME_BUCKETS)
        self.assertIn("Long", READING_TIME_BUCKETS)

    def test_reading_time_buckets_conditions(self):
        """Test that READING_TIME_BUCKETS conditions work correctly"""
        short_condition, _ = READING_TIME_BUCKETS["Very_Short"]
        medium_condition, _ = READING_TIME_BUCKETS["Medium"]
        long_condition, _ = READING_TIME_BUCKETS["Long"]
        
        self.assertTrue(short_condition(1))
        self.assertTrue(medium_condition(7))
        self.assertTrue(long_condition(15))

    def test_engagement_buckets_structure(self):
        """Test that ENGAGEMENT_BUCKETS has correct structure"""
        self.assertIsInstance(ENGAGEMENT_BUCKETS, dict)
        self.assertIn("Low", ENGAGEMENT_BUCKETS)
        self.assertIn("Medium", ENGAGEMENT_BUCKETS)
        self.assertIn("High", ENGAGEMENT_BUCKETS)
        self.assertIn("Very_High", ENGAGEMENT_BUCKETS)

    def test_engagement_buckets_conditions(self):
        """Test that ENGAGEMENT_BUCKETS conditions work correctly"""
        low_condition, _ = ENGAGEMENT_BUCKETS["Low"]
        medium_condition, _ = ENGAGEMENT_BUCKETS["Medium"]
        high_condition, _ = ENGAGEMENT_BUCKETS["High"]
        very_high_condition, _ = ENGAGEMENT_BUCKETS["Very_High"]
        
        self.assertTrue(low_condition(20))
        self.assertTrue(medium_condition(40))
        self.assertTrue(high_condition(75))
        self.assertTrue(very_high_condition(150))

    def test_retention_buckets_structure(self):
        """Test that RETENTION_BUCKETS has correct structure"""
        self.assertIsInstance(RETENTION_BUCKETS, dict)
        self.assertIn("Low_Completion", RETENTION_BUCKETS)
        self.assertIn("Moderate_Completion", RETENTION_BUCKETS)
        self.assertIn("High_Completion", RETENTION_BUCKETS)

    def test_retention_buckets_conditions(self):
        """Test that RETENTION_BUCKETS conditions work correctly"""
        low_condition, _ = RETENTION_BUCKETS["Low_Completion"]
        moderate_condition, _ = RETENTION_BUCKETS["Moderate_Completion"]
        high_condition, _ = RETENTION_BUCKETS["High_Completion"]
        
        self.assertTrue(low_condition(0.3))
        self.assertTrue(moderate_condition(0.65))
        self.assertTrue(high_condition(0.85))

    def test_stv_values_valid_ranges(self):
        """Test that STV constant values are in valid range [0, 1]"""
        stv_constants = [DETERMINISTIC_STV, AI_FAILURE_STV, UNKNOWN_STV]
        
        for stv in stv_constants:
            self.assertIsInstance(stv, tuple)
            self.assertEqual(len(stv), 2)
            strength, confidence = stv
            self.assertGreaterEqual(strength, 0.0)
            self.assertLessEqual(strength, 1.0)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_stv_values_specific(self):
        """Test specific STV values"""
        # Deterministic should be high confidence
        self.assertEqual(DETERMINISTIC_STV, (1.0, 1.0))
        
        # AI failure and unknown should be moderate
        self.assertEqual(AI_FAILURE_STV, (0.5, 0.5))
        self.assertEqual(UNKNOWN_STV, (0.5, 0.5))

    def test_all_bucket_stv_ranges_valid(self):
        """Test all STV ranges in buckets are valid"""
        all_buckets = [
            LENGTH_BUCKETS, READING_TIME_BUCKETS, 
            ENGAGEMENT_BUCKETS, RETENTION_BUCKETS
        ]
        
        for bucket_dict in all_buckets:
            for label, (condition, stv_range) in bucket_dict.items():
                strength, confidence = stv_range
                self.assertGreaterEqual(strength, 0.0, f"Invalid strength in {label}")
                self.assertLessEqual(strength, 1.0, f"Invalid strength in {label}")
                self.assertGreaterEqual(confidence, 0.0, f"Invalid confidence in {label}")
                self.assertLessEqual(confidence, 1.0, f"Invalid confidence in {label}")

    def test_bucket_stv_ranges_ordered(self):
        """Test that STV ranges within buckets are ordered (min < max)"""
        all_buckets = [
            LENGTH_BUCKETS, READING_TIME_BUCKETS, 
            ENGAGEMENT_BUCKETS, RETENTION_BUCKETS
        ]
        
        for bucket_dict in all_buckets:
            for label, (condition, stv_range) in bucket_dict.items():
                strength_min, strength_max = stv_range
                self.assertLessEqual(strength_min, strength_max, 
                                   f"STV range not ordered in {label}")

    def test_mindplex_api_configuration(self):
        """Test Mindplex API configuration"""
        self.assertIsInstance(MINDPLEX_API_DOMAIN, str)
        self.assertTrue(MINDPLEX_API_DOMAIN.startswith("https://"))
        
        self.assertIsInstance(USER_ARTICLES_ENDPOINT_TEMPLATE, str)
        self.assertIn("{username}", USER_ARTICLES_ENDPOINT_TEMPLATE)
        self.assertIn("{page}", USER_ARTICLES_ENDPOINT_TEMPLATE)

    def test_default_username(self):
        """Test default username is set"""
        self.assertIsInstance(DEFAULT_USERNAME, str)
        self.assertTrue(len(DEFAULT_USERNAME) > 0)

    def test_default_headers(self):
        """Test default headers structure"""
        self.assertIsInstance(DEFAULT_HEADERS, dict)
        self.assertIn("User-Agent", DEFAULT_HEADERS)
        self.assertIn("Accept", DEFAULT_HEADERS)
        self.assertEqual(DEFAULT_HEADERS["Accept"], "application/json")

    def test_analysis_prompt_template(self):
        """Test analysis prompt template"""
        self.assertIsInstance(ANALYSIS_PROMPT_TEMPLATE, str)
        self.assertIn("{content_snippet}", ANALYSIS_PROMPT_TEMPLATE)
        self.assertIn("tone", ANALYSIS_PROMPT_TEMPLATE.lower())
        self.assertIn("sentiment", ANALYSIS_PROMPT_TEMPLATE.lower())


class TestBucketBoundaries(unittest.TestCase):
    """Test bucket boundary conditions"""

    def test_length_bucket_boundaries(self):
        """Test LENGTH_BUCKETS at exact boundary values"""
        short, (short_cond, _) = ("Short", LENGTH_BUCKETS["Short"])
        medium, (medium_cond, _) = ("Medium", LENGTH_BUCKETS["Medium"])
        long, (long_cond, _) = ("Long", LENGTH_BUCKETS["Long"])
        
        # Test exact boundaries
        self.assertTrue(short_cond(499))
        self.assertTrue(medium_cond(500))
        self.assertTrue(medium_cond(1500))
        self.assertTrue(long_cond(1501))

    def test_reading_time_bucket_boundaries(self):
        """Test READING_TIME_BUCKETS at exact boundary values"""
        very_short, (vs_cond, _) = ("Very_Short", READING_TIME_BUCKETS["Very_Short"])
        short, (s_cond, _) = ("Short", READING_TIME_BUCKETS["Short"])
        medium, (m_cond, _) = ("Medium", READING_TIME_BUCKETS["Medium"])
        long, (l_cond, _) = ("Long", READING_TIME_BUCKETS["Long"])
        
        # Test boundary transitions
        self.assertTrue(vs_cond(1))
        self.assertTrue(s_cond(2))
        self.assertTrue(m_cond(5))
        self.assertTrue(l_cond(11))

    def test_engagement_bucket_boundaries(self):
        """Test ENGAGEMENT_BUCKETS at exact boundary values"""
        low, (low_cond, _) = ("Low", ENGAGEMENT_BUCKETS["Low"])
        medium, (med_cond, _) = ("Medium", ENGAGEMENT_BUCKETS["Medium"])
        high, (high_cond, _) = ("High", ENGAGEMENT_BUCKETS["High"])
        very_high, (vh_cond, _) = ("Very_High", ENGAGEMENT_BUCKETS["Very_High"])
        
        self.assertTrue(low_cond(29))
        self.assertTrue(med_cond(30))
        self.assertTrue(med_cond(49))
        self.assertTrue(high_cond(50))
        self.assertTrue(high_cond(100))
        self.assertTrue(vh_cond(101))

    def test_retention_bucket_boundaries(self):
        """Test RETENTION_BUCKETS at exact boundary values"""
        low, (low_cond, _) = ("Low_Completion", RETENTION_BUCKETS["Low_Completion"])
        moderate, (mod_cond, _) = ("Moderate_Completion", RETENTION_BUCKETS["Moderate_Completion"])
        high, (high_cond, _) = ("High_Completion", RETENTION_BUCKETS["High_Completion"])
        
        self.assertTrue(low_cond(0.49))
        self.assertTrue(mod_cond(0.50))
        self.assertTrue(mod_cond(0.80))
        self.assertTrue(high_cond(0.81))


class TestConfigConsistency(unittest.TestCase):
    """Test consistency across configuration"""

    def test_all_buckets_have_consistent_stv_structure(self):
        """Test that all buckets follow consistent STV structure"""
        all_buckets = [
            LENGTH_BUCKETS, READING_TIME_BUCKETS, 
            ENGAGEMENT_BUCKETS, RETENTION_BUCKETS
        ]
        
        for bucket_dict in all_buckets:
            # All buckets should have at least one label
            self.assertGreater(len(bucket_dict), 0)
            
            # All entries should have consistent tuple structure
            for label, (condition, stv_range) in bucket_dict.items():
                self.assertIsInstance(stv_range, (tuple, list))
                self.assertEqual(len(stv_range), 2)

    def test_stv_constant_types(self):
        """Test that STV constants are tuples with float values"""
        for stv_const, name in [
            (DETERMINISTIC_STV, "DETERMINISTIC_STV"),
            (AI_FAILURE_STV, "AI_FAILURE_STV"),
            (UNKNOWN_STV, "UNKNOWN_STV"),
        ]:
            self.assertIsInstance(stv_const, tuple, f"{name} is not a tuple")
            self.assertEqual(len(stv_const), 2, f"{name} does not have 2 elements")
            strength, confidence = stv_const
            self.assertIsInstance(strength, float, f"{name} strength is not float")
            self.assertIsInstance(confidence, float, f"{name} confidence is not float")


if __name__ == "__main__":
    unittest.main()
