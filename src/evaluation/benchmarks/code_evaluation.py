"""
Code Generation Benchmarks

Evaluates model on standard coding benchmarks:
- HumanEval: 164 hand-written programming problems
- MBPP: Python programming problems
- Custom DevTasks: Real-world development scenarios
"""

import json
import logging
import subprocess
import tempfile
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import ast
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of running a single test"""
    task_id: str
    passed: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0


class HumanEvalBenchmark:
    """
    HumanEval benchmark for code generation
    Tests functional correctness of generated code
    """

    def __init__(self, dataset_path: str = "data/benchmarks/humaneval.jsonl"):
        self.dataset_path = Path(dataset_path)
        self.problems = self._load_problems()

    def _load_problems(self) -> List[Dict]:
        """Load HumanEval problems"""
        if not self.dataset_path.exists():
            logger.warning(f"HumanEval dataset not found at {self.dataset_path}")
            return self._create_sample_problems()

        problems = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                problems.append(json.loads(line))

        logger.info(f"Loaded {len(problems)} HumanEval problems")
        return problems

    def _create_sample_problems(self) -> List[Dict]:
        """Create sample problems for demonstration"""
        return [
            {
                "task_id": "HumanEval/0",
                "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
                "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
                "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n\ncheck(has_close_elements)"
            },
            {
                "task_id": "HumanEval/1",
                "prompt": "def separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
                "canonical_solution": "    result = []\n    current_string = []\n    current_depth = 0\n\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string.clear()\n\n    return result\n",
                "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n\ncheck(separate_paren_groups)"
            }
        ]

    def execute_code(self, code: str, test: str, timeout: int = 5) -> Tuple[bool, Optional[str]]:
        """
        Execute code with test cases

        Args:
            code: Generated code
            test: Test code
            timeout: Execution timeout in seconds

        Returns:
            Tuple of (passed, error_message)
        """
        # Combine code and test
        full_code = f"""
from typing import List, Tuple, Optional
import math

{code}

{test}
"""

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name

        try:
            # Execute code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Check if execution succeeded
            if result.returncode == 0:
                return True, None
            else:
                error = result.stderr or result.stdout
                return False, error

        except subprocess.TimeoutExpired:
            return False, "Execution timeout"
        except Exception as e:
            return False, str(e)
        finally:
            # Clean up
            Path(temp_file).unlink(missing_ok=True)

    def evaluate_solution(self, task_id: str, solution: str) -> TestResult:
        """
        Evaluate a single solution

        Args:
            task_id: Task identifier
            solution: Generated code solution

        Returns:
            TestResult
        """
        # Find problem
        problem = next((p for p in self.problems if p["task_id"] == task_id), None)
        if not problem:
            return TestResult(
                task_id=task_id,
                passed=False,
                error_message="Task not found"
            )

        # Execute with tests
        passed, error = self.execute_code(solution, problem["test"])

        return TestResult(
            task_id=task_id,
            passed=passed,
            error_message=error
        )

    def evaluate_model(self, model, tokenizer, max_problems: int = 10) -> Dict:
        """
        Evaluate model on HumanEval benchmark

        Args:
            model: Language model
            tokenizer: Tokenizer
            max_problems: Maximum problems to evaluate

        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating on {max_problems} HumanEval problems...")

        results = []
        problems_to_test = self.problems[:max_problems]

        for problem in problems_to_test:
            task_id = problem["task_id"]
            prompt = problem["prompt"]

            # Generate solution
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=512,
                    temperature=0.2,
                    do_sample=False  # Greedy for consistency
                )
                solution = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Evaluate
                result = self.evaluate_solution(task_id, solution)
                results.append(result)

                logger.info(f"{task_id}: {'✓' if result.passed else '✗'}")

            except Exception as e:
                logger.error(f"Error evaluating {task_id}: {e}")
                results.append(TestResult(
                    task_id=task_id,
                    passed=False,
                    error_message=str(e)
                ))

        # Compute metrics
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        pass_at_1 = passed_count / total_count if total_count > 0 else 0

        metrics = {
            "benchmark": "HumanEval",
            "total_problems": total_count,
            "passed": passed_count,
            "failed": total_count - passed_count,
            "pass@1": pass_at_1,
            "results": [{"task_id": r.task_id, "passed": r.passed, "error": r.error_message} for r in results]
        }

        logger.info(f"\nHumanEval Results:")
        logger.info(f"Pass@1: {pass_at_1:.2%} ({passed_count}/{total_count})")

        return metrics


class CustomDevTasksBenchmark:
    """
    Custom benchmark for real-world development tasks
    """

    def __init__(self):
        self.tasks = self._create_dev_tasks()

    def _create_dev_tasks(self) -> List[Dict]:
        """Create custom development tasks"""
        return [
            {
                "task_id": "DevTask/1",
                "category": "API Design",
                "description": "Create a RESTful API endpoint for user authentication",
                "prompt": "Write a Flask API endpoint that handles user login with email and password, returns a JWT token on success, and proper error handling.",
                "criteria": [
                    "Uses POST method",
                    "Validates input",
                    "Returns JWT token",
                    "Handles errors",
                    "Uses bcrypt for password hashing"
                ]
            },
            {
                "task_id": "DevTask/2",
                "category": "Database Query",
                "description": "Write efficient database query with joins",
                "prompt": "Write a SQL query to fetch all orders with customer details and product information, including customers who haven't placed orders yet.",
                "criteria": [
                    "Uses LEFT JOIN",
                    "Includes all customers",
                    "Retrieves product details",
                    "Efficient query structure"
                ]
            },
            {
                "task_id": "DevTask/3",
                "category": "Algorithm",
                "description": "Implement rate limiting",
                "prompt": "Implement a rate limiter class that allows a maximum of N requests per time window, using the sliding window algorithm.",
                "criteria": [
                    "Sliding window implementation",
                    "Thread-safe",
                    "Efficient time complexity",
                    "Proper cleanup of old requests"
                ]
            }
        ]

    def evaluate_task(self, task_id: str, solution: str) -> Dict:
        """
        Evaluate a custom dev task

        Args:
            task_id: Task identifier
            solution: Generated solution

        Returns:
            Evaluation result dictionary
        """
        task = next((t for t in self.tasks if t["task_id"] == task_id), None)
        if not task:
            return {"error": "Task not found"}

        # Automated evaluation based on criteria
        criteria_met = []
        for criterion in task["criteria"]:
            # Simple keyword/pattern matching
            # In production, would use more sophisticated analysis
            met = any(
                keyword.lower() in solution.lower()
                for keyword in criterion.split()[:3]  # Check first few words
            )
            criteria_met.append({
                "criterion": criterion,
                "met": met
            })

        score = sum(c["met"] for c in criteria_met) / len(criteria_met)

        return {
            "task_id": task_id,
            "category": task["category"],
            "score": score,
            "criteria_results": criteria_met
        }


def run_comprehensive_evaluation(model, tokenizer, output_file: str = "results/evaluation.json"):
    """
    Run comprehensive evaluation across all benchmarks

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer
        output_file: Output file for results
    """
    logger.info("Running comprehensive evaluation...")

    results = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "benchmarks": {}
    }

    # HumanEval
    humaneval = HumanEvalBenchmark()
    results["benchmarks"]["humaneval"] = humaneval.evaluate_model(model, tokenizer, max_problems=10)

    # Custom Dev Tasks
    dev_tasks = CustomDevTasksBenchmark()
    results["benchmarks"]["dev_tasks"] = {
        "total_tasks": len(dev_tasks.tasks),
        "tasks": dev_tasks.tasks
    }

    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dumps(results, f, indent=2)

    logger.info(f"Evaluation complete. Results saved to {output_file}")

    return results


if __name__ == "__main__":
    # Example: Create sample benchmark
    benchmark = HumanEvalBenchmark()

    # Test a sample solution
    solution = """
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i, num1 in enumerate(numbers):
        for j, num2 in enumerate(numbers):
            if i != j and abs(num1 - num2) < threshold:
                return True
    return False
"""

    result = benchmark.evaluate_solution("HumanEval/0", solution)
    print(f"Result: {result.passed}")
    if not result.passed:
        print(f"Error: {result.error_message}")
