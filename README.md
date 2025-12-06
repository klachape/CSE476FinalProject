# CSE476FinalProject
Final Project for CSE476: Implement an inference-time agent to solve reasoning requests


# Run as default
After cloning the repository, run the generate_answer_template.py script to replicate my results in the cse_476_final_project_answers.json file.


# Run with custom input and output paths
If you would like to test the agent using different questions, change generate_answer_template.py INPUT_PATH to a new JSON file with the same format and number of questions as cse_476_final_test_data.json. 
If you would like to change the location of the answers, change the OUTPUT_PATH variable in generate_answer_template.py to your desired location before running the script.

You can then run generate_answer_template.py to generate new answers in cse_476_final_project_answers.json or the new answers file path.

# Note:
My answers did not come out as expected. The agent often answered with far too long of an answer instead of simple, single-worded answers.
I would have liked to have spent more time tuning the prompts to get answers in the correct format, however, to run the agent a single time
took around 5 hours and I do not have time to do that again before the deadline.