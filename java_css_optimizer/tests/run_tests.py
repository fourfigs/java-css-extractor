import unittest
import sys
import time
import logging
from pathlib import Path
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_tests():
    """Run all tests with detailed reporting."""
    start_time = time.time()
    logger = logging.getLogger(__name__)
    
    # Discover and load tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        start_dir=str(Path(__file__).parent),
        pattern='test_*.py'
    )
    
    # Create test runner with detailed output
    test_runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout
    )
    
    # Run tests
    logger.info("Starting test suite...")
    result = test_runner.run(test_suite)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Log results
    logger.info("\nTest Results Summary:")
    logger.info(f"Total Tests: {result.testsRun}")
    logger.info(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"Failed: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Execution Time: {execution_time:.2f} seconds")
    
    # Log failures
    if result.failures:
        logger.error("\nFailures:")
        for failure in result.failures:
            logger.error(f"\n{failure[0]}")
            logger.error(failure[1])
    
    # Log errors
    if result.errors:
        logger.error("\nErrors:")
        for error in result.errors:
            logger.error(f"\n{error[0]}")
            logger.error(error[1])
    
    # Save detailed report
    report_path = Path('test_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Test Report - {datetime.now()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Tests: {result.testsRun}\n")
        f.write(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}\n")
        f.write(f"Failed: {len(result.failures)}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
        
        if result.failures:
            f.write("Failures:\n")
            f.write("-" * 50 + "\n")
            for failure in result.failures:
                f.write(f"\n{failure[0]}\n")
                f.write(failure[1])
                f.write("\n")
        
        if result.errors:
            f.write("\nErrors:\n")
            f.write("-" * 50 + "\n")
            for error in result.errors:
                f.write(f"\n{error[0]}\n")
                f.write(error[1])
                f.write("\n")
    
    return len(result.failures) + len(result.errors)

if __name__ == '__main__':
    sys.exit(run_tests()) 