Press `ctrl` + `shift` + `V` in VS Code to view a (much nicer) preview of this Markdown file.

# ASO Project Template

This ZIP archive is your template for all coding projects in this course.

It contains numerous files, but you will only have to edit one of them: `src/aso/optimiser.py`

Feel free to look into the other files too if you are interested, but you don't have to. We structured this template like a Python package named `aso` to give you an idea of what this could look like in a professional environment.

Before you submit your work, please read the section [Project Submission](#project-submission) at the end of this document.

## Prerequisites

### Install Python

Ensure that Python 3.10 or later is installed on your machine. You can download an installer [here](https://www.python.org/downloads/).

### Install uv

We use the package manager uv by Astral. To install uv, follow [these steps](https://docs.astral.sh/uv/getting-started/installation/). On Windows, running
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
in Command Prompt is usually all you have to do.

You can check your installation by typing
```
uv -V
```
in any terminal on your machine. If you see something like
```
uv 0.8.23 (00d3aa378 2025-10-04)
```
you're good to go.

### Create a Virtual Environment and Install Dependencies

Open this directory in VS Code.

In the terminal, run
```
uv venv
```
to create a new virtual environment. A virtual environment is an isolated workspace that contains its own Python interpreter and packages, preventing conflicts with other projects on your system.

Activate the virtual environment with
```
.venv\Scripts\activate
```
on Windows or
```
source .venv/bin/activate
```
on Linux and macOS.

If this does not work, you may have to override your execution policy as follows:
- Type "Windows PowerShell" in the Windows search bar
- Right-click and select "Run as Administrator"
- Confirm with "Yes"
- Paste and run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Confirm with `Y`

Then, run
```
uv sync
```
to install all Python packages used in the code.

If there are import errors auch as `Import "numpy" could not be resolved`, ensure that the correct Python interpreter is selected. To do this,
- open a Python file,
- click on the Python version number in the lower right corner of the VS Code window,
- select the correct interpreter at the top of the screen.

This could be Python 3.13.7 (aso), for example. The path should be `.\.venv\Scripts\python.exe`.

## Project Work

For the project work, we deleted some functions from the source code, so you can implement and test them yourself.

The function signatures and docstrings already give you an idea of what the final function could look like. You do not have to use all arguments in the signature (you can ignore all the `callback` parameters, for example).

No additional imports are required for the projects. Please do not use packages that are not part of the Python standard library as we would have to install them to test your code.

### First Project

Implement the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) and its analytic gradient in `aso.problem_factory.ProblemFactory.rosenbrock`.

Then, implement a convergence check in `aso.optimiser.Optimiser.converged` and a steepest descent algorithm in `aso.optimiser.Optimiser.steepest_descent`. You can use your convergence check in your steepest descent algorithm.

Ensure that your virtual environment is activated and run
```
pytest -m project_1
```

This will apply your algorithm to seven test problems. A test is passed if your steepest descent implementation converges to a minimum in less than 100,000 iterations and failed otherwise.

#### Note on Implementation

The tests expect the `Optimiser` to modify the design variables in place. Hence, you have to update them using one of the following options.

In place, efficient, intuitive:
```
self.x += step_size * search_direction
```

Equivalent but less intuitive:
```
numpy.add(self.x, step_size * search_direction, out=self.x)
```

Less efficient because it creates a new, temporary array for the right-hand side, copies it back to the left-hand side, and discards the temporary right-hand side:
```
self.x[:] = self.x + step_size * search_direction
```

The following option, on the other hand, would not pass the tests, since `x` will be pointing to a different location in memory after each iteration:
```
self.x = self.x + step_size * search_direction
```

#### Note on Debugging

By default, `pytest` intercepts `sys.stdout`, so you cannot `print` to the console during tests. If, however, you want to use `print` statements for debugging in combination with `pytest`, just add the `-s` flag to the `pytest` command. It's a shortcut for `--capture=no`, which disables all [capturing](https://docs.pytest.org/en/latest/how-to/capture-stdout-stderr.html), so your `print` statements will be visible on the console again.

Alternatively, you can write log messages, which will be saved to the logs directory. For example,
```
logger.debug(f"iteration = {iteration}, x = {self.x}")
```

You can also use the debugging features of your IDE. There is a Python Debugger extension by Microsoft for VS Code, for example.

#### Hint

If your algorithm does not pass some or even all of the tests, the problem could be the algorithm itself, but it could also be a parameter such as the step size, especially if it is constant. Try different values to find one that is suitable for all seven test problems.

### Second Project

Implement a line search algorithm in `aso.optimiser.Optimiser.line_search` and a constrained SQP algorithm in `aso.optimiser.Optimiser.sqp_constrained`. Your SQP algorithm should
- update the inverse Hessian approximation using the L-BFGS formula,
- handle side constraints by variable projection and gradient masking.
You can use your line search algorithm and your convergence check from the previous project in your constrained SQP algorithm.

As before, test your algorithm by running
```
pytest -m project_2
```

The test problems are constrained versions of the unconstrained problems from the first project. The step size is now limited to 1,000.

If you compare the log files from the SQP runs (`pytest -m project_2`) to those from the steepest descent runs (`pytest -m project_1`), what do you see?

It may also be interesting to compare the performance of
- steepest descent with a fixed step size vs. steepest descent with line search,
- SQP with a fixed step size vs. SQP with line search,
- steepest descent with line search vs. SQP with line search.

### Third Project

Implement your MMA algorithm in `aso.optimiser.Optimiser.mma`.

Again, you can reuse the convergence check and the line search from the previous two submissions.

As before, test your algorithm by running
```
pytest -m project_3
```

This time, there is only one test with an iteration limit of 1,000.

### Fourth Project

Your last submission will not be a Python project, but something very practical. Stay tuned!

## Project Submission

Ensure that your virtual environment is activated, navigate to the root directory of your project (aso_project_template if you did not rename it), and run
```
python scripts/zip_project.py --config scripts/zip_project.yaml --project x --student xxxxxxxx
```
where `x` is the number of the submission (`1`, `2`, or `3`) and `xxxxxxxx` is your eight-digit TUM student ID.

Then, upload the generated ZIP archive (aso_project_x_student_xxxxxxxx.zip) on Moodle. It will only contain the file you edited.
