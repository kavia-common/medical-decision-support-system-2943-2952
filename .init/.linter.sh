#!/bin/bash
cd /home/kavia/workspace/code-generation/medical-decision-support-system-2943-2952/medical_decision_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

