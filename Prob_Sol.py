import os.path
import time
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import json

# Fixed model initialization
model_prob_sol = OllamaLLM(model="gpt-oss:latest", temperature=0, keep_alive="1s", options={"nothink": True})

template_background = """You are a patent analyst. Based on background of the invention, draft a paragraph on problem that is being addressed by the patent. Additionally, identify lines from the background that correspond to exact verbatim of the problem drafted in the patent.
Output the background paragraph and the verbatim text from the patent in a json format having fields as "problem" and "problem_verbatim". The field "problem" should include the paragraph describing the problem, while the field "problem_verbatim" should include the exact verbatim identified from the background drafted in the patent.
Here is the background from patent: {BACKGROUND}
"""

template_solution = """You are a patent analyst. Based on claims of the invention and the background, draft a paragraph on solution claimed in the patent. Additionally, identify lines from the claims that correspond to exact verbatim of the solution claimed in the patent.
Output the solution paragraph and the verbatim text from the patent in a json format having fields as "solution" and "solution_verbatim". The field "solution" should include the paragraph describing the solution, while the field "solution_verbatim" should include the exact verbatim identified from the claims drafted in the patent.
Here is the background of patent: {BACKGROUND}
Here is the claims of the patent: {CLAIMS}
"""

prompt_background = ChatPromptTemplate.from_template(template_background)
prompt_claims = ChatPromptTemplate.from_template(template_solution)

problem_chain = prompt_background | model_prob_sol
solution_chain = prompt_claims | model_prob_sol

file_path = os.path.join("Data", "g_patents_scanner_printer_merged_text.csv")
pat_df = pd.read_csv(file_path)

# Define output file path
output_file = os.path.join("Data", "g_patents_with_analysis.csv")

# Check if output file already exists
existing_data = None
if os.path.exists(output_file):
    print(f"Output file {output_file} already exists. Loading existing data...")
    try:
        existing_data = pd.read_csv(output_file)
        print(f"Loaded existing data with {len(existing_data)} records")
        # Create a set of patent_ids that already exist in output file for quick lookup
        existing_patent_ids = set(existing_data['patent_id'].dropna().astype(str))
        print(f"Found {len(existing_patent_ids)} existing patent IDs")
    except Exception as e:
        print(f"Error loading existing data: {e}")
        existing_patent_ids = set()
else:
    existing_patent_ids = set()

# Process each row and extract JSON data
results = []

for index, row in pat_df.iterrows():
    try:
        background_text = row['summary_text']
        claim_text = row['merged_claim_text']
        patent_id = row.get('patent_id', None)

        # Check if texts are valid
        if not isinstance(background_text, str) or not background_text.strip():
            print(f"Skipping row {index}: Invalid background text")
            continue

        if not isinstance(claim_text, str) or not claim_text.strip():
            print(f"Skipping row {index}: Invalid claim text")
            continue

        # Check if patent_id exists in output file
        if patent_id is not None and str(patent_id) in existing_patent_ids:
            print(f"Skipping row {index} (patent ID {patent_id} already exists in output file)")
            continue

        # Get results from LLM only if patent_id doesn't exist in output file
        print(f"Processing row {index} for patent ID: {patent_id}")
        result_background = problem_chain.invoke({"BACKGROUND": background_text})
        result_solution = solution_chain.invoke({"BACKGROUND": background_text, "CLAIMS": claim_text})

        # Parse JSON responses
        try:
            # Extract JSON from response (in case it's wrapped in markdown or quotes)
            if isinstance(result_background, str):
                # Look for JSON content between markers
                json_start = result_background.find('{')
                json_end = result_background.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = result_background[json_start:json_end]
                    background_json = json.loads(json_str)
                else:
                    print(f"Could not extract JSON from background response for row {index}")
                    background_json = {}
            else:
                background_json = result_background if isinstance(result_background, dict) else {}

        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing background JSON for row {index}: {e}")
            background_json = {}

        try:
            # Extract JSON from solution response
            if isinstance(result_solution, str):
                json_start = result_solution.find('{')
                json_end = result_solution.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = result_solution[json_start:json_end]
                    solution_json = json.loads(json_str)
                else:
                    print(f"Could not extract JSON from solution response for row {index}")
                    solution_json = {}
            else:
                solution_json = result_solution if isinstance(result_solution, dict) else {}

        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing solution JSON for row {index}: {e}")
            solution_json = {}

        # Combine the results
        combined_result = {
            'index': index,
            'patent_id': patent_id  # Using .get() for safer access
        }

        # Add background data
        if 'problem' in background_json:
            combined_result['problem'] = background_json['problem']
        if 'problem_verbatim' in background_json:
            combined_result['problem_verbatim'] = background_json['problem_verbatim']

        # Add solution data
        if 'solution' in solution_json:
            combined_result['solution'] = solution_json['solution']
        if 'solution_verbatim' in solution_json:
            combined_result['solution_verbatim'] = solution_json['solution_verbatim']

        # Append to output file immediately after processing each row
        result_df = pd.DataFrame([combined_result])

        if existing_data is not None and len(existing_data) > 0:
            # Check if this patent_id already exists in existing data
            existing_patent_ids.add(str(patent_id))  # Add the new patent ID to our tracking set
            # Append to existing file
            result_df.to_csv(output_file, mode='a', header=False, index=False)
            print(f"Appended row {index} to {output_file}")
        else:
            # Create new file with first record
            result_df.to_csv(output_file, mode='w', header=True, index=False)
            print(f"Created {output_file} with first record")
            existing_patent_ids.add(str(patent_id))  # Track the patent ID

        existing_data = pd.read_csv(output_file) if os.path.exists(output_file) else None

        print(f"Processed and saved row {index}")
        time.sleep(0.5)  # Small delay to prevent overwhelming the model

    except Exception as e:
        print(f"Error processing row {index}: {str(e)}")
        continue

print("Processing completed.")
