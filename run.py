import base64
import logging
from openai import OpenAI
import tempfile
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import re
import json
import sys
import argparse
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from metrics.metrics import (
    BleuCalculator,
    CodeBleuCalculator,
    RougeLCalculator,
    EditDistanceCalculator,
    ChrFCalculator,
    CrystalBLEUCalculator
)

def clean_code(code):

    import re
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)
    lines = []
    for line in code.split('\n'):
        line = re.sub(r'#.*$', '', line)
        if line.strip():
            lines.append(line)
    
    cleaned_code = '\n'.join(lines)
    cleaned_code = re.sub(r'\n\s*\n', '\n', cleaned_code)
    cleaned_code = cleaned_code.strip()
    
    return cleaned_code



def calculate_metrics(reference, hypothesis):
    
    metrics = {}
    
    try:
        bleu_calc = BleuCalculator()
        codebleu_calc = CodeBleuCalculator(lang='python')
        rougel_calc = RougeLCalculator()
        edit_calc = EditDistanceCalculator()
        chrf_calc = ChrFCalculator()
        crystal_bleu_calc = CrystalBLEUCalculator()
        
        metrics['bleu'] = bleu_calc.calculate(reference, hypothesis)
        metrics['codebleu'] = codebleu_calc.calculate(reference, hypothesis)
        metrics['rougel'] = rougel_calc.calculate(reference, hypothesis)
        metrics['edit_distance'] = edit_calc.calculate(reference, hypothesis)
        metrics['chrf'] = chrf_calc.calculate(reference, hypothesis)
        metrics['crystal_bleu'] = crystal_bleu_calc.calculate(reference, hypothesis)
      
            
    except Exception as e:
        print(f"error: {e}")
        return {}
    
    return metrics



def parsing_agent(image_path=None, max_retries=3,args=None):
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base
    )

    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
    except Exception as e:
        logger.error(f"Error reading image file: {str(e)}")
        return None


    system_prompt = """You are a architecture analyst responsible for structural analysis and dependency extraction of the given model architecture diagram. Your task is to:
    1. Parse the diagram to identify key architectural elements, including modules, submodules, functional blocks, mathematical symbols, and any associated textual annotations.
    2. Extract both explicit and implicit structural relationships, such as intermodule connections, data flow paths, hierarchical structures, and execution dependencies.
    3. Determine the logical execution order of the components and clarify how different parts of the model interact at the structural level.
    4. Organize the extracted information into a structured analysis that captures:
        • All components and their roles.
        • Connection types (e.g., sequential, parallel, conditional).
        • Execution logic and dependency flow.
    5. Format your output as a clear, structured analysis that can be used as input for downstream code generation.
    6. Do not include any meta-comments, explanations, or reasoning steps—output only the final analysis.
    """

    user_prompt = """Please analyze the model architecture shown in the image and provide a detailed breakdown of its structure and component relationships."""


    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                    }
                }
            ]
        }
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0.3,
            )
            
            analysis_result = response.choices[0].message.content.strip()
            logger.info(f"Successfully analyzed model architecture on attempt {attempt + 1}")
            return analysis_result
                
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                continue
    
    logger.error(f"All {max_retries} attempts failed")
    return None


def generation_agent(analysis_result=None, image_path=None, ground_truth=None, imports_string = "",max_retries=3,args=None):
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base
    )

    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
    except Exception as e:
        logger.error(f"Error reading image file: {str(e)}")
        return None, 0.0


    system_prompt = """You are a Python code generator specialized in creating multi-file implementations of model architectures based on research paper diagrams. Your tasks are:
    1. Analyze the provided model diagram image and the provided model structure analysis carefully.
    2. Generate complete, functional Python code for the model architecture distributed across the relevant .py files based on the model structure analysis.
    3. For each file, follow the provided file-specific information (file name, required imported packages, and predefined class names) and implement not only the class definitions but also the complete inner functionalities (methods, forward passes, etc.) as dictated by the diagram.
    4. Ensure that each class is fully implemented without leaving any method or functionality incomplete.
    5. Only provide raw Python code with clear file separators (e.g., comment headers indicating the file name) without any explanations, markdown formatting, or additional text.
    6. Do not provide examples."""
    
    user_prompt = f"""Implement the model architecture as depicted in the image. If the model spans multiple files, generate the corresponding Python code for each file separately.
    Focus solely on constructing the complete model structure with fully implemented functionalities inside each class (including necessary methods, forward functions, etc.) as specified by the diagram. Do not include code for datasets, training, or testing.
    Model Structure Analysis:
    {analysis_result}
    Required File-Specific Information:
    {imports_string}
    """


    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                    }
                }
            ]
        }
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0.3,
            )
            
            generated_code = response.choices[0].message.content.strip()
            logger.info(f"Successfully generated code on attempt {attempt + 1}")

            generated_code = clean_code(generated_code)
            generated_code = re.sub(r'^```\w*\n', '', generated_code)  
            generated_code = re.sub(r'```$', '', generated_code) 
            ground_truth = clean_code(ground_truth)
            metrics = calculate_metrics(ground_truth, generated_code)
        
            return generated_code, metrics
                
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                continue
    
    logger.error(f"All {max_retries} attempts failed")
    return None, 0.0





def check_agent(code_to_check, image_path=None,ground_truth=None,imports_string = "", max_retries=3,args=None):
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base
    )


    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
    except Exception as e:
        logger.error(f"Error reading image file: {str(e)}")
        return None

    system_prompt = """You are a model architecture verification expert tasked with verifying and refining the generated code based on the given model architecture and the architecture analysis. Your task is to:
    1.Carefully analyze the provided model architecture diagram and the corresponding structured analysis.
    2.Compare this architecture with the given Python code implementation to determine whether the code fully and correctly reflects the structural components, execution logic, and data flow described in the architecture.
    3.If the implementation perfectly matches the architecture in terms of components, structure, and execution semantics, respond with only: yes.
    4.If the implementation is incomplete, inconsistent, or incorrect, output a complete, corrected version of the Python code that accurately reflects the architecture. The corrected code should:
        • Include all necessary components and connections.
        • Preserve the logical execution order inferred from the architecture analysis.
    5.Do not include any explanations, reasoning steps, or comments—your response must consist only of the final, verified code or the word yes.
    """

    user_prompt = f"""Please analyze whether the following Python implementation code matches the model architecture shown in the image.

    Implementation Code:
    {code_to_check}

    Required File-Specific Information:
    {imports_string}

    If the implementation code perfectly matches the model architecture, respond with exactly "yes".
    If not, provide ONLY the corrected implementation code."""

   
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                    }
                }
            ]
        }
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0.3,
            )
            
            generated_code = response.choices[0].message.content.strip()
            logger.info(f"Successfully generated code on attempt {attempt + 1}")
           
            if generated_code == "yes":
                final_code = code_to_check
            else:
                final_code = generated_code

            final_code = clean_code(final_code)
            final_code = re.sub(r'^```\w*\n', '', final_code)  
            final_code = re.sub(r'```$', '', final_code) 
            ground_truth = clean_code(ground_truth)
            metrics = calculate_metrics(ground_truth, final_code)
            
            return final_code, metrics
                
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                continue
    
    logger.error(f"All {max_retries} attempts failed")
    return None,0.0





def load_dataset(args):

    dataset_root = args.dataset_root
    
    dataset = []
    
    for domain in os.listdir(dataset_root):
        domain_path = os.path.join(dataset_root, domain)
        if not os.path.isdir(domain_path):
            continue

        
        for paper in os.listdir(domain_path):
            if not paper[0].isdigit():
                  continue
                  
            image_dir = os.path.join(domain_path, paper, "model.png")
            if not os.path.exists(image_dir):
                  image_dir = os.path.join(domain_path, paper, "model.jpg")
            if not os.path.exists(image_dir):
                  image_dir = os.path.join(domain_path, paper, "model.jpeg")
            if not os.path.exists(image_dir):
                  image_dir = os.path.join(domain_path, paper, "model.PNG")
            code_path = os.path.join(domain_path, paper, "llm_model_filger.json")
            
            if not os.path.exists(image_dir) or not os.path.exists(code_path):
                code_path = os.path.join(domain_path, paper, "llm_model.json")
                

            dataset.append((image_dir, code_path))
                  

    
    return dataset



def extract_imports(code):
    import_lines = ""
    lines = code.split('\n')
    current_import = ""
    in_parentheses = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('import ') or line.startswith('from '):
            if current_import:  
                import_lines+=current_import +"\n"
            current_import = line
            if '(' in line:
                in_parentheses = True
            elif ')' in line:
                in_parentheses = False
                import_lines+=current_import +"\n"
                current_import = ""

        elif in_parentheses:
            current_import += " " + line.strip('(),')
            if ')' in line:
                in_parentheses = False
                import_lines+=current_import +"\n"
                current_import = ""
    
    if current_import:
        import_lines+=current_import +"\n"
    
    return import_lines


def extract_class_definitions(code):
    import re
    class_pattern = r'class\s+\w+\s*(?:\([^)]*\))?\s*:'
    class_matches = re.finditer(class_pattern, code)

    class_lines = []
    for match in class_matches:
        class_line = match.group()
        class_lines.append(class_line)
    
    return '\n'.join(class_lines)


def parse_args():
    parser = argparse.ArgumentParser(description='Model Generation and Evaluation')
    
    parser.add_argument('--output_path', type=str, default='',
                      help='Path to save the generation results')
    parser.add_argument('--output_path_continue', type=str, default='',
                      help='Path to save the generation results')
    parser.add_argument('--prompt_strategy', type=str, default='package_class',
                      help='Prompt strategy: package, package_class')
    parser.add_argument('--agent_prompt_strategy', type=str, default='test1',
                      help='Prompt strategy: test1, raw')
    
    parser.add_argument('--api_key', type=str, 
                      default='',
                      help='OpenAI API key')
    parser.add_argument('--api_base', type=str,
                      default='',
                      help='API base URL')
    parser.add_argument('--model', type=str, default='',
                      help='Model to use for generation')
    parser.add_argument('--temperature', type=float, default=0.3,
                      help='Temperature for generation')
    
    args = parser.parse_args()
    return args


def save_result(paper_key, result_dict, output_path):
    try:
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        all_results[paper_key] = result_dict[paper_key] 
        

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved for paper: {paper_key}")
    except Exception as e:
        logger.error(f"Error saving results for {paper_key}: {e}")


if __name__ == "__main__":
   
    args = parse_args()
    with open('dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    all_papers_num = len(dataset)
    all_results = {}
    
    total_metrics = {
        'bleu': 0.0,
        'codebleu': 0.0,
        'rougel': 0.0,
        'edit_distance': 0.0,
        'chrf': 0.0,
        'crystal_bleu': 0.0
    }

    gen_agent_metrics = {
        'bleu': 0.0,
        'codebleu': 0.0,
        'rougel': 0.0,
        'edit_distance': 0.0,
        'chrf': 0.0,
        'crystal_bleu': 0.0
    }

    check_agent_metrics = {
        'bleu': 0.0,
        'codebleu': 0.0,
        'rougel': 0.0,
        'edit_distance': 0.0,
        'chrf': 0.0,
        'crystal_bleu': 0.0
    }

    count = 0
    API_KEYS = []

    for index,key in enumerate(API_KEYS):
        if os.path.exists(args.output_path_continue):
            with open(args.output_path_continue, 'r', encoding='utf-8') as f:
                output_continue = json.load(f)
        else:
            output_continue = {}
        
            with open(args.output_path_continue, 'w', encoding='utf-8') as f:
                json.dump(output_continue, f, indent=2, ensure_ascii=False)
            logger.info(f"Created new output continue file: {args.output_path_continue}")
            
     
        if all_papers_num == len(output_continue):
            break
        if args.api_base == "":
            args.api_key = key
        fail_count = 0
        for value in tqdm(dataset, desc="Processing papers", unit="paper"):
            image_path = value[0]
            ground_truth_path = value[1]

       
            domain = os.path.basename(os.path.dirname(os.path.dirname(ground_truth_path)))
            paper_id = os.path.basename(os.path.dirname(ground_truth_path))
            paper_key = f"{domain}/{paper_id}"

            if paper_key in output_continue:
                continue
            
           

            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                ground_truth_dict = json.load(f)

            imports_string = ""
            for file_path, code in ground_truth_dict.items():
                imports_string+="File:"+file_path+":\n"
                imports = extract_imports(code)
                if imports:  
                    imports_string+=imports +"\n"

                if args.prompt_strategy == "package_class":
                    class_definitions = extract_class_definitions(code)
                    if class_definitions:
                        imports_string+=class_definitions +"\n"
                    imports_string += "\n"
            
            ground_truth = '\n'.join(f"{value}" for key, value in ground_truth_dict.items())

            try:
                analysis = parsing_agent(image_path,args=args)
                if analysis:
                    generation_code, generation_metrics = generation_agent(analysis, image_path,ground_truth,imports_string = imports_string,args=args)
                    check_code , check_metrics = check_agent(generation_code, image_path,ground_truth,imports_string = imports_string,args=args)
                    
                if not isinstance(check_metrics, dict):
                    fail_count += 1
                    if fail_count >= 50:
                        break
                    continue
                    
                if check_code is None:
                    fail_count += 1
                    if fail_count >= 50:
                        break
                    continue
                
                all_results[paper_key] = {
                    "analysis": analysis,
                    "ground_truth": ground_truth_dict,
                    "generation_code": generation_code,
                    "check_code": check_code,
                    "generation_metrics": generation_metrics,
                    "check_metrics": check_metrics
                }

                save_result(paper_key, all_results, args.output_path_continue)
                    
            except Exception as e:
                continue
            
            for metric_name, score in agent2_metrics.items():
                if metric_name == "codebleu":
                    gen_agent_metrics[metric_name] += score["codebleu"]
                else:
                    gen_agent_metrics[metric_name] += score

            for metric_name, score in agent3_metrics.items():
                if metric_name == "codebleu":
                    check_agent_metrics[metric_name] += score["codebleu"]
                else:
                    check_agent_metrics[metric_name] += score


        if args.api_base != "":
            break
           

    try:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    except Exception as e:

    if count > 0:
        for metric_name, total in total_metrics.items():
            avg_score = total / count
            print(f"Average {metric_name.upper()} Score: {avg_score:.4f}")

 
        for metric_name, total in all_agent2_metrics.items():
            avg_score = total / count
            print(f"Average {metric_name.upper()} Score: {avg_score:.4f}")

        for metric_name, total in all_agent3_metrics.items():
            avg_score = total / count
            print(f"Average {metric_name.upper()} Score: {avg_score:.4f}")

   
        
