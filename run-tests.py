import subprocess
import pathlib
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define ANSI escape codes for colors
RESET = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"


def extract_and_print(result, path, idx) -> bool:
    output = result.stdout if result.returncode == 0 else result.stderr
    extracted = output.replace("[()]\n", "")
    has_failure = "Error" in extracted
    if not has_failure:
        extracted = "test passed"
    status_color = RED if has_failure else GREEN
    print(YELLOW + f"Test {idx + 1}: {path}" + RESET)
    print(status_color + extracted + RESET)
    print(YELLOW + f"Exit-code: {result.returncode}" + RESET)
    print("-" * 40)
    return has_failure

def run_test_file(test_file):
    try:
        result = subprocess.run(
            [metta_run_command, str(test_file)],
            capture_output=True,
            text=True,
            check=True,
        )
        return result, test_file, False
    except subprocess.CalledProcessError as e:
        return e, test_file, True

def print_ascii_art(text):
    art = f"""
                {text}
           """
    print(CYAN + art + RESET)

metta_run_command = "metta"
root = pathlib.Path("../")
testMettaFiles = list(root.rglob("*test.metta"))
total_files = len(testMettaFiles)
results = []
fails = 0
print_ascii_art("Parallel Test Runner")
with ThreadPoolExecutor() as executor:
    future_to_test = {
        executor.submit(run_test_file, test_file): idx
        for idx, test_file in enumerate(testMettaFiles)
    }
    for future in as_completed(future_to_test):
        idx = future_to_test[future]
        try:
            result, path, has_failure = future.result()
            if isinstance(result, subprocess.CalledProcessError):
                print(RED + f"Error with {path}: {result.stderr}" + RESET)
                fails += 1
                continue
            has_failure = extract_and_print(result, path, idx)
            if has_failure:
                fails += 1
        except Exception as exc:
            print(RED + f"Test {idx + 1}: generated an exception: {exc}" + RESET)
            fails += 1
print(CYAN + "\nTest Summary" + RESET)
print(f"{total_files} files tested.")
print(RED + f"{fails} failed." + RESET)
print(GREEN + f"{total_files - fails} succeeded." + RESET)
if fails > 0:
    print(RED + "Tests failed. Process Exiting with exit code 1" + RESET)
    sys.exit(1)
