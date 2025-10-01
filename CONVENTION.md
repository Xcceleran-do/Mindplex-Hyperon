## MeTTa Formatting Rules

Since MeTTa lacks a flexible formatter, formatting is done as you write code. Some basic formatting conventions for MeTTa include:

### Function Calls
- If the parameters are short or few, keep the function call on a single line.
- If the parameters are long or numerous, the function name should stick with the left-most bracket, and the parameters should be placed on the next lines with increased indentation.

```metta
;; Example function with short parameter
(foo $x)

;; Example function with long parameters
(foo 
    (and (== $x $y) (< $x 100))
    (X is valid parameter)
    (X isn't valid parameter)
)

;; Example function with many parameters
(let* 
    (
        ($x (someFunction1 ...))
        ($y (someFunction2 ...))
        ($z (someFunction3 ...))
    ) 
    ($x $y $z)
)
```

- Whenever calling the `if` function, the function call should be a one line if the `$then` and `$else` cases are simple. But if either one of them is complex, the function name and the condition should be on the same line as the opening bracket and the rest of the parameters should be on the next line with the same level of indentation as the condition.

```metta
;; Calling if with simple $then and $else conditions
(if (== $x 1) True False)

;; Calling if with complex $then and $else conditions
(if (== $x 1)
    True
    (let* 
        (
            ($x' (> $x 0))
            ($result (< $x 10))
        )
        (and $x' $result)
    )
)
```

- Multi-line functions: The closing bracket of a function should match the indentation level of its opening bracket.
- Indentation: Increase indentation level for nested parameters or function calls for better readability.

### Function Definitions
- When creating multiple definitions of a function, don't add empty lines between the definitions unless the definitions are complex. For complex multi-line definitions, add a single line between them for clarity.
- No empty lines should exist between a function and its type signature or its documentation.
- When writing complex function definitions, the name of the function and its parameters should be on the same line and the rest of the function definition should start on a new line with the same level of indentation as the function name.

```metta
;; Example function that does nothing with simple definitions.
(: foo (-> Atom Atom))
(= (foo $x) ($x))
(= (foo $x $x) ($x))
(= (foo $x $y) ($x $y))
(= (foo $x $y $z) ($x $y $z))

;; Example function with complex definitions. 
(: foo' (-> Atom Atom))
(= (foo $x) 
   (if (...)
       (...)
       (...)
   )
)

(= (foo $x $y)
   (let $bar (if (== $x $y) (...) (...)))
)
```

## Commenting Guidelines
- Always use double semicolons (`;;`) for comments instead of a single semicolon (`;`) to maintain consistency.

## Documentation Standards
- Every new function should be documented using comments right before its definition.
- Comments should be sufficient to describe the function's purpose and how it works at a high level.
- Include examples in the comments if necessary.
- It is recommended to comment every function, no matter how small it seems.
# Naming and Code Conventions for Mindplex Hyperon

## File Naming
- All files and folders should use camelCase (e.g., `userProfile.metta`, `recommendationEngine.metta`, `userProfile/`).
- Test files must be named with a `-test` suffix and use camelCase (e.g., `userProfile-test.metta`).

## Function Naming
- Use camelCase for all function names (e.g., `getUserHistory`, `predictInteraction`).

## Variable Naming
- Use camelCase for all variable names (e.g., `userHistory`, `interactionScore`).
## Commit Message Conventions

A helpful technique is to use a heading and bullet points:

- The first line of the commit message should be a short summary of all changes (e.g., `feat: implement admin middleware`, `fix: update vulnerable package`, `chore: clean up codebase`).
- After the summary, add a list of bullet points, one for every file you changed. If you made similar changes across multiple files, you can group them in a single bullet point.
- Start each bullet point with CRUD-based language: `- added ...`, `- revised ...`, `- removed ...`, etc.
- Always include the WHY in your explanation: `- added aria-label attribute to the icon tag since a label wasn’t previously provided and we need to pass accessibility standards.`
- Reference issues or PRs when relevant (e.g., `fix bug in userHistory #42`).

For more information on writing commit messages, see [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
## Other Conventions
- Keep commits focused; avoid mixing unrelated changes in a single commit.
- Use descriptive branch names (e.g., `feature/userProfile`, `bugfix/recommendationEngine`).

## General Conventions
- All new features must include corresponding test files in the appropriate `tests` subfolder.
- Code should be modular and readable, with clear comments for complex logic.
- Avoid abbreviations unless they are widely recognized.
- Use descriptive names for all files, functions, and variables.
- Follow the MeTTa language best practices for syntax and structure.

## Pull Request Requirements
- Every new PR must include tests for new features or changes.
- PRs should follow the naming conventions outlined above.
- PRs must be confirmed by the contributor before merging.

---
For more details, see the `CONTRIBUTING.md` file.
