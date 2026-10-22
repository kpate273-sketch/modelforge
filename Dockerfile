# Containerization (Module 6: Deploying Models Safely).
# The image pins the exact Python version, so the project runs identically on a
# laptop, a teammate's machine and a server. There are no third-party packages to
# install — the standard library is the whole dependency list.
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Fail the build if the tests fail: a broken image never gets built.
RUN python tests/test_all.py

# Default: run the full lab and write the HTML report.
CMD ["python", "-m", "modelforge", "lab", "--out", "/app/report.html"]
