import os
import re
from typing import TYPE_CHECKING, Dict, List, Union
import requests
import json
from swift.utils import get_dist_setting
import uuid
import math
from typing import List, Union
import requests
if TYPE_CHECKING:
    from swift.llm import InferRequest

import time
import numpy as np


rank, local_rank, world_size, local_world_size = get_dist_setting()


class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify==0.5.2'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # edge case
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class GeoAnswerFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        """
        奖励函数：检查回答是否包含<answer>标签并且内容符合地理位置格式。
        支持中文分号和英文分号，允许两段（国家，城市）或三段（国家，城市，具体位置）。
        """
        results = []
        for content in completions:
            content = content.strip()
            
            # 首先提取<answer>标签中的内容
            import re
            answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL)
            if not answer_match:
                results.append(0.0)
                continue
                
            answer_content = answer_match.group(1).strip()
            
            # 检查answer内容是否符合格式：支持2段或3段，用逗号分隔
            # 支持中文逗号（，）和英文逗号（,）
            parts = re.split(r'[;]', answer_content)
            parts = [part.strip() for part in parts if part.strip()]  # 去除空白部分
            
            # 检查是否有2段或3段有效内容
            if len(parts) >= 2 and len(parts) <= 3:
                results.append(1.0)
            else:
                results.append(0.0)
                
        return results



        
class ThinkAnswerConsistency(ORM):
    def __init__(self, 
                 model_url: str = "http://localhost:8000/v1/chat/completions",
                 model_name: str = "YOUR_MODEL_NAME",
                 timeout: int = 30,
                 country_weight: float = 0.1,
                 region_weight: float = 0.6,
                 location_weight: float = 0.3,
                 min_lengths=90,
                 max_lengths=200):
        """
        Consistency reward function for chain-of-thought and answer (including country, region, and precise stages).

        Args:
            model_url: URL address for the large model API
            model_name: Model name/path
            timeout: Request timeout (seconds)
            country_weight: Weight for the country stage
            region_weight: Weight for the region-guessing stage
            location_weight: Weight for the precise localization stage
            min_lengths: Minimum length for reasoning
            max_lengths: Maximum length for reasoning
        """
        self.model_url = model_url
        self.model_name = model_name
        self.timeout = timeout
        self.country_weight = country_weight
        self.region_weight = region_weight
        self.location_weight = location_weight
        self.min_lengths = min_lengths
        self.max_lengths = max_lengths
        self.system_prompt = """
        You are an expert in geographic location inference.
        Task: Based on the given reasoning, infer the geographical location conclusion according to the current inference stage.
        You will receive input in JSON format:
        {
        "id": "Request unique identifier ",
        "stage": "Current inference stage (CountryIdentification/RegionalGuess/PreciseLocalization)",
        "Reasoning": "Reasoning of the current stage",
        "Previous conclusion": {
            "CountryIdentification": "Country conclusion (if any)",
            "RegionalGuess": "Region Conclusion (if any)"
        }
        }
        Please deduce a conclusion based on this information and output it in JSON format:
        {
        "id": "The id as echoed back",
        "Conclusion": "Your reasoning conclusion"
        }
        Note:
        1. It is necessary to strictly output JSON format without any other text
        2. The received ID must be displayed as the original sample in the output
        3. The conclusion should be concise and clear. The CountryIdentification only includes the name of the country, the RegionalGuess may be a province or city, and PreciseLocalization is further positioning
        4. Use pre conclusions to assist in the current stage of reasoning
        5. When guessing the region, the conclusion of CountryIdentification should be considered
        6. When accurately positioning, both CountryIdentification and RegionalGuess conclusions should be considered simultaneously
        """

    def _extract_think_content(self, completion: str) -> dict:
        """
        Extract JSON content of chain-of-thought from model output.

        Args:
            completion: Full text generated by the model

        Returns:
            Parsed chain-of-thought dictionary, or None if parsing fails
        """
        try:
            import re, json
            think_match = re.search(r'<think>(.*?)</think>', completion, flags=re.S | re.I)
            if think_match:
                think_text = think_match.group(1).strip()
            else:
                think_text = completion

            json_match = re.search(r'\{.*\}', think_text, flags=re.S)
            if not json_match:
                print(f"No JSON block found, original content: {think_text[:200]}...")
                return None

            json_str = json_match.group(0)
            think_json = json.loads(json_str)

            if "ChainOfThought" not in think_json:
                print(f"ChainOfThought not found, original content: {think_text[:200]}...")
                return None

            chain = think_json["ChainOfThought"]
            if not isinstance(chain, dict):
                print(f"chain type error, original content: {think_text[:200]}...")
                return None

            required_steps = ["CountryIdentification", "RegionalGuess", "PreciseLocalization"]
            for step in required_steps:
                if step not in chain:
                    print(f"step not found, original content: {think_text[:200]}...")
                    return None

                step_data = chain[step]
                if not isinstance(step_data, dict):
                    print(f"step_data type error, original content: {think_text[:200]}...")
                    return None

                required_fields = ["Clues", "Reasoning", "Conclusion"]
                for field in required_fields:
                    if field not in step_data:
                        print(f"field not found, original content: {think_text[:200]}...")
                        return None

            return think_json

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Parsing error: {e}")
            return None

    def _call_model_for_reasoning(self, basis: str, stage: str,
                                  country_conclusion: str = None,
                                  region_conclusion: str = None) -> str:
        """
        Call the large model to perform reasoning and get a conclusion based on clues and reasoning basis.

        Args:
            basis: Reasoning basis
            stage: Current inference stage (CountryIdentification/RegionalGuess/PreciseLocalization)
            country_conclusion: Conclusion from the country stage (optional)
            region_conclusion: Conclusion from the region stage (optional)

        Returns:
            The inferred conclusion, or None if failed
        """
        import requests, uuid, json
        try:
            sample_id = str(uuid.uuid4())

            previous_conclusions = {}
            if country_conclusion is not None:
                previous_conclusions["CountryIdentification"] = str(country_conclusion)
            if region_conclusion is not None:
                previous_conclusions["RegionalGuess"] = str(region_conclusion)

            input_data = {
                "id": sample_id,
                "stage": str(stage),
                "Reasoning": str(basis)
            }

            if previous_conclusions:
                input_data["Previous conclusion"] = previous_conclusions

            user_input = json.dumps(input_data, ensure_ascii=False)

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": 512,
                "temperature": 0.1,
                "enable_thinking": False
            }

            response = requests.post(
                self.model_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )

            if response.status_code != 200:
                return None

            data = response.json()
            if "choices" not in data or len(data["choices"]) == 0:
                return None

            model_output = data["choices"][0]["message"]["content"].strip()
            return self._parse_conclusion(model_output, expected_id=sample_id)

        except Exception as e:
            print(f"[ThinkAnswerConsistency] Error while calling model: {e}")
            return None

    def _parse_conclusion(self, model_output: str, expected_id: str) -> str:
        """
        Parse the conclusion from large model output and verify ID consistency.

        Args:
            model_output: Output text from the model
            expected_id: Expected ID value

        Returns:
            Extracted conclusion, or None on failure or ID mismatch
        """
        import json, re
        try:
            json_match = re.search(r'\{.*\}', model_output, flags=re.S)
            if not json_match:
                print(f"[ThinkAnswerConsistency] No JSON block found, original content: {model_output[:200]}...")
                return None

            output_json = json.loads(json_match.group(0))

            if not isinstance(output_json, dict):
                print(f"[ThinkAnswerConsistency] Type error, original content: {model_output[:200]}...")
                return None

            if "id" not in output_json or "Conclusion" not in output_json:
                print(f"[ThinkAnswerConsistency] Key missing, original content: {model_output[:200]}...")
                return None

            if output_json["id"] != expected_id:
                print(f"[ThinkAnswerConsistency] id mismatch, expected: {expected_id}, actual: {output_json['id']}")
                return None

            return str(output_json["Conclusion"]).strip()

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[ThinkAnswerConsistency] Parse error: {e}, original content: {model_output[:200]}...")
            return None

    def _calculate_stage_reward(self, original_conclusion: str, model_conclusion: str) -> float:
        """
        Calculate the reward for a single stage (exact match required).

        Args:
            original_conclusion: Original conclusion
            model_conclusion: Conclusion inferred by model

        Returns:
            Reward value: 1.0 if exactly equal, otherwise 0.0
        """
        if original_conclusion is None or model_conclusion is None:
            return 0.0

        orig_clean = str(original_conclusion).strip().replace(' ', '')
        model_clean = str(model_conclusion).strip().replace(' ', '')

        return 1.0 if orig_clean == model_clean else 0.0

    def __call__(self, completions: list, solution=None, **kwargs) -> list:
        """
        Compute chain-of-thought and answer consistency rewards.

        Args:
            completions: List of responses generated by the model
            solution: Gold answers (not used here)
            **kwargs: Extra arguments

        Returns:
            List[float]: List of reward scores
        """
        import time, random
        rewards = []

        for completion in completions:
            try:
                think_json = self._extract_think_content(completion)
                if think_json is None:
                    rewards.append(0.0)
                    continue

                chain = think_json["ChainOfThought"]

                stage_rewards = []
                stage_length_weight = []
                stage_names = ["CountryIdentification", "RegionalGuess", "PreciseLocalization"]

                country_original_conclusion = chain["CountryIdentification"]["Conclusion"]
                region_original_conclusion = chain["RegionalGuess"]["Conclusion"] if "RegionalGuess" in chain else None

                # CountryIdentification stage
                stage_data = chain["CountryIdentification"]
                basis = stage_data.get("Reasoning", "")
                original_conclusion = stage_data.get("Conclusion", "")
                model_conclusion = self._call_model_for_reasoning(
                    basis, "CountryIdentification"
                )
                country_stage_reward = self._calculate_stage_reward(original_conclusion, model_conclusion)
                stage_rewards.append(country_stage_reward)

                basis_length = len(basis)
                if basis_length < self.min_lengths:
                    length_weight = 0.0
                elif basis_length > self.max_lengths:
                    length_weight = 1.0
                else:
                    import math
                    normalized_length = (basis_length - self.min_lengths) / (self.max_lengths - self.min_lengths)
                    x = (normalized_length - 0.5) * 8
                    length_weight = 1 / (1 + math.exp(-x))
                stage_length_weight.append(length_weight)
                time.sleep(random.uniform(0.01, 0.05))

                # RegionalGuess stage
                stage_data = chain["RegionalGuess"]
                basis = stage_data.get("Reasoning", "")
                original_conclusion = stage_data.get("Conclusion", "")
                model_conclusion = self._call_model_for_reasoning(
                    basis, "RegionalGuess", country_conclusion=country_original_conclusion
                )
                region_stage_reward = self._calculate_stage_reward(original_conclusion, model_conclusion)
                stage_rewards.append(region_stage_reward)
                basis_length = len(basis)
                if basis_length < self.min_lengths:
                    length_weight = 0.0
                elif basis_length > self.max_lengths:
                    length_weight = 1.0
                else:
                    import math
                    normalized_length = (basis_length - self.min_lengths) / (self.max_lengths - self.min_lengths)
                    x = (normalized_length - 0.5) * 8
                    length_weight = 1 / (1 + math.exp(-x))
                stage_length_weight.append(length_weight)
                time.sleep(random.uniform(0.01, 0.05))

                # PreciseLocalization stage
                stage_data = chain["PreciseLocalization"]
                basis = stage_data.get("Reasoning", "")
                original_conclusion = stage_data.get("Conclusion", "")
                model_conclusion = self._call_model_for_reasoning(
                    basis, "PreciseLocalization",
                    country_conclusion=country_original_conclusion,
                    region_conclusion=chain["RegionalGuess"]["Conclusion"]
                )
                location_stage_reward = self._calculate_stage_reward(original_conclusion, model_conclusion)
                stage_rewards.append(location_stage_reward)
                basis_length = len(basis)
                if basis_length < self.min_lengths:
                    length_weight = 0.0
                elif basis_length > self.max_lengths:
                    length_weight = 1.0
                else:
                    import math
                    normalized_length = (basis_length - self.min_lengths) / (self.max_lengths - self.min_lengths)
                    x = (normalized_length - 0.5) * 8
                    length_weight = 1 / (1 + math.exp(-x))
                stage_length_weight.append(length_weight)
                time.sleep(random.uniform(0.01, 0.05))

                # Weighted sum according to stage weights
                final_reward = (
                    self.country_weight * stage_length_weight[0] * stage_rewards[0]
                    + self.region_weight * stage_length_weight[1] * stage_rewards[1]
                    + self.location_weight * stage_length_weight[2] * stage_rewards[2]
                )
                rewards.append(float(final_reward))

            except Exception as e:
                print(f"[ThinkAnswerConsistency] Error processing sample: {e}")
                rewards.append(0.0)

        return rewards
class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 tokenizer=None,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.tokenizer = tokenizer
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []
        for content, acc_reward in zip(completions, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(self.tokenizer.encode(content))
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, tokenizer, soft_max_length, soft_cache_length):
        self.tokenizer = tokenizer
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


    
class GeoScoreAccuracy(ORM):
    """GeoScore accuracy reward function based on OpenCage API (with cache)"""

    def __init__(
        self,
        api_keys: List[str] = [
            "YOUR_API_KEYS"
        ], # The number of api keys should be equal to the number of GPUs
        max_distance: float = 2000.0, # 10 * tau
        confidence_threshold: int = 2,
        timeout: int = 10,
        cache_file: str = "YOUR_CACHE_FILE",
    ):
        """
        Initialize the GeoScore accuracy reward function.

        Args:
            api_keys: OpenCage API key list (multi-GPU support)
            max_distance: Maximum distance threshold (km) for GeoScore calculation
            confidence_threshold: Confidence threshold (1-10); below this returns 0, recommended 3-7
            timeout: API request timeout (seconds)
        """
        self.api_list = api_keys
        self.max_distance = max_distance
        self.confidence_threshold = confidence_threshold
        self.timeout = timeout
        self.base_url = "https://api.opencagedata.com/geocode/v1/json"
        self.query_count = 0
        self.daily_limit = 100000
        self.max_qps = 15
        self.current_gpu = self._get_current_gpu()
        self.api_key = api_keys[self.current_gpu]
        self.cache = self._load_cache(cache_file)
        print(f"GPU{self.current_gpu},AK={self.api_key}")

    def _load_cache(self, cache_file: str):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_answer_content(self, completion: str) -> str:
        """
        Extract content within the <answer> tag from model output.

        Args:
            completion: Full model-generated text

        Returns:
            Extracted answer content
        """
        if completion is None:
            return None
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.S | re.I)
        if m:
            return m.group(1).strip()
        try:
            json_match = re.search(r"\{.*\}", completion, flags=re.S)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                if "FinalAnswer" in data:
                    return str(data["FinalAnswer"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return None

    def _smart_delay(self):
        import random
        import time

        delay = random.uniform(0, 1)
        time.sleep(delay)

    def _get_current_gpu(self):
        """
        Get the actual GPU index (0~7) used by the current process in single-node multi-GPU setting.
        """
        import os
        import torch
        import torch.distributed as dist

        local_rank = os.environ.get("LOCAL_RANK", None)
        if local_rank is not None:
            return int(local_rank)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            num_gpus = torch.cuda.device_count()
            return rank % num_gpus
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return -1

    def _call_opencage_geocoding(
        self, address: str, retry_count: int = 0
    ) -> Dict:
        """
        Call the OpenCage geocoding API.

        Args:
            address: Address to geocode
            retry_count: Number of retries

        Returns:
            Dictionary with coordinates and confidence info
        """
        self.query_count += 1
        if self.query_count % 500 == 0:
            print(
                f"[GeoScoreAccuracy3][GPU:{self.current_gpu}] Query count: {self.query_count}"
            )
        if self.query_count > self.daily_limit:
            error_result = {
                "status": "error",
                "message": "Exceeded daily API call limit",
            }
            print(f"[GeoScoreAccuracy3] Exceeded daily API call limit")
            return error_result
        self._smart_delay()
        params = {
            "q": address.replace(";", ",").replace("；", ",").replace("，", ","),
            "key": self.api_key,
            "language": "en",
            "limit": 1,
        }
        try:
            response = requests.get(
                self.base_url, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            if "rate" in data and "remaining" in data["rate"]:
                remaining = data["rate"]["remaining"]
                if (remaining % 100) == 0 or (remaining < 20):
                    print(
                        f"[GeoScoreAccuracy3] OpenCage API remaining quota {remaining},API={self.api_key}"
                    )
            if data["status"]["code"] == 200:
                results = data["results"]
                if len(results) == 0:
                    print(
                        f"[GeoScoreAccuracy3] No results found, message:{data['status']['message']}, address:{address}"
                    )
                    address_parts = address.split(";")
                    print(
                        f"[GeoScoreAccuracy3] address_parts: {address_parts},len:{len(address_parts)}"
                    )
                    if len(address_parts) > 1:
                        shorter_address = ";".join(address_parts[:-1])
                        print(
                            f"[GeoScoreAccuracy3] No result, retrying with shorter address {shorter_address}"
                        )
                        return self._call_opencage_geocoding(
                            shorter_address, retry_count
                        )
                    return {
                        "status": "error",
                        "code": 200,
                        "message": data["status"]["message"],
                        "address": address,
                    }
                else:
                    results = results[0]
                    coordinates = results["geometry"]
                    lat = coordinates["lat"]
                    lng = coordinates["lng"]
                    components = results.get("components", {})
                    formatted_address = results.get("formatted", "")
                    confidence = results.get("confidence", 0)
                    api_result = {
                        "status": "success",
                        "longitude": lng,
                        "latitude": lat,
                        "confidence": confidence,
                        "formatted_address": formatted_address,
                        "components": components,
                    }
                    return api_result
            if data["status"]["code"] == 429:
                print(f"[GeoScoreAccuracy3] Rate limit: {data['status']['message']}")
                if retry_count < 5:
                    print(
                        f"[GeoScoreAccuracy3] Rate limit, retry {retry_count+1}"
                    )
                    time.sleep(0.1)
                    return self._call_opencage_geocoding(address, retry_count + 1)
                else:
                    return {
                        "status": "error",
                        "code": 429,
                        "message": data["status"]["message"],
                        "address": address,
                    }
            if data["status"]["code"] == 410:
                print(
                    f"[GeoScoreAccuracy3] Address too long: message:{data['status']['message']}, address:{address}"
                )
                address_parts = address.split(";")
                if len(address_parts) > 1:
                    shorter_address = ";".join(address_parts[:-1])
                    return self._call_opencage_geocoding(
                        shorter_address, retry_count
                    )
                else:
                    return {
                        "status": "error",
                        "code": 410,
                        "message": data["status"]["message"],
                        "address": address,
                    }
            if data["status"]["code"] == 408:
                print(
                    f"[GeoScoreAccuracy3] Timeout: message:{data['status']['message']}, address:{address}"
                )
                if retry_count < 5:
                    print(
                        f"[GeoScoreAccuracy3] Timeout, retry {retry_count+1}"
                    )
                    time.sleep(0.1)
                    return self._call_opencage_geocoding(address, retry_count + 1)
                else:
                    return {
                        "status": "error",
                        "code": 408,
                        "message": data["status"]["message"],
                        "address": address,
                    }
            if data["status"]["code"] == 403:
                print(
                    f"[GeoScoreAccuracy3] Forbidden: message:{data['status']['message']}, address:{address}"
                )
                return {
                    "status": "error",
                    "code": 403,
                    "message": data["status"]["message"],
                    "address": address,
                }
            else:
                print(
                    f"[GeoScoreAccuracy3] OpenCage API unknown error: {data['status']}"
                )
                return {
                    "status": "error",
                    "code": data["status"]["code"],
                    "message": data["status"]["message"],
                    "address": address,
                }
        except requests.exceptions.RequestException as e:
            print(f"[GeoScoreAccuracy] OpenCage API request error: {e}")
            if retry_count < 5:
                print(
                    f"[GeoScoreAccuracy3] OpenCage API request error, retry {retry_count+1}"
                )
                time.sleep(0.1)
                return self._call_opencage_geocoding(address, retry_count + 1)
            else:
                return {
                    "status": "error",
                    "code": 0,
                    "message": str(e),
                    "address": address,
                }
        except Exception as e:
            return {
                "status": "error",
                "code": 999,
                "message": str(e),
                "address": address,
            }

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate the distance (in kilometers) between two latitude/longitude points using the Haversine formula.

        Args:
            lat1, lon1: Latitude and longitude of the first point
            lat2, lon2: Latitude and longitude of the second point

        Returns:
            Distance in kilometers between the two points
        """
        lat1, lon1, lat2, lon2 = map(
            math.radians, [lat1, lon1, lat2, lon2]
        )
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        return c * r

    def _calculate_geoscore(self, distance: float) -> float:
        """
        Calculate GeoScore from distance.

        Args:
            distance: Distance in kilometers

        Returns:
            GeoScore value
        """
        if distance >= self.max_distance:
            return 0.0
        return math.exp(-10 * (distance / self.max_distance))

    def __call__(
        self,
        completions: List[str],
        solution: Union[str, List[str]],
        **kwargs,
    ) -> List[float]:
        """
        Compute GeoScore-based accuracy rewards using API.

        Args:
            completions: List of model-generated answers
            solution: List of ground truth answers or a single answer
            **kwargs: Additional arguments

        Returns:
            List[float]: List of reward scores
        """
        rewards = []

        if isinstance(solution, str):
            solutions = [solution] * len(completions)
        elif isinstance(solution, list):
            solutions = solution
        else:
            print(
                f"[GeoScoreAccuracy] Unsupported solution type: {type(solution)}, setting all rewards to 0"
            )
            return [0.0] * len(completions)
        if len(solutions) == 1 and len(completions) > 1:
            solutions = solutions * len(completions)
        elif len(solutions) != len(completions):
            print(
                f"[GeoScoreAccuracy] completions({len(completions)}) and solutions({len(solutions)}) length mismatch, setting all rewards to 0"
            )
            return [0.0] * len(completions)

        for completion, sol in zip(completions, solutions):
            try:
                pred_answer = self._extract_answer_content(completion)
                gt_answer = self._extract_answer_content(str(sol))

                if not pred_answer or not gt_answer:
                    rewards.append(0.0)
                    continue

                pred_result = self._call_opencage_geocoding(pred_answer)
                if pred_result.get("status") != "success":
                    rewards.append(0.0)
                    print(
                        f"[GeoScoreAccuracy3] Prediction failed: {pred_answer}, reward=0"
                    )
                    continue

                pred_confidence = pred_result.get("confidence", 0)
                pred_lng = pred_result.get("longitude")
                pred_lat = pred_result.get("latitude")

                if gt_answer.replace(" ", "") in self.cache:
                    gt_result = self.cache[gt_answer.replace(" ", "")]
                else:
                    print(
                        f"[GeoScoreAccuracy3] type(gt_answer): {type(gt_answer)}"
                    )
                    print(
                        f"[GeoScoreAccuracy3] Cache miss! Querying address: {gt_answer}, no-space address: {gt_answer.replace(' ','')}"
                    )
                    gt_result = self._call_opencage_geocoding(gt_answer)
                if gt_result.get("status") != "success":
                    rewards.append(0.0)
                    print(
                        f"[GeoScoreAccuracy3] Ground truth failed: {gt_answer}, reward=0"
                    )
                    continue

                gt_confidence = gt_result.get("confidence", 0)
                gt_lng = gt_result.get("longitude")
                gt_lat = gt_result.get("latitude")

                if (
                    pred_lng is None
                    or pred_lat is None
                    or gt_lng is None
                    or gt_lat is None
                ):
                    rewards.append(0.0)
                    print(
                        f"[GeoScoreAccuracy] Invalid coordinates: {pred_lng}, {pred_lat}, {gt_lng}, {gt_lat}, reward=0"
                    )
                    continue

                distance = self._calculate_distance(
                    pred_lat, pred_lng, gt_lat, gt_lng
                )
                geoscore = self._calculate_geoscore(distance)

                rewards.append(float(geoscore))

            except Exception as e:
                print(f"[GeoScoreAccuracy] Error occurred: {e}")
                rewards.append(0.0)

        return rewards


class TripleFieldAccuracy(ORM):
    """Geographical accuracy reward function based on triple segmentation (cosine similarity + threshold for all segments)."""
    
    def __init__(self, 
                 field1_weight: float = 0.1, 
                 field2_weight: float = 0.6, 
                 field3_weight: float = 0.3,
                 field1_threshold: float = 0.7,
                 field2_threshold: float = 0.5,
                 split_new: bool = True,
                 text2vec_server_url: str = "http://localhost:5001"):
        """
        Initialize the triple field accuracy reward function (cosine similarity version).
        
        Args:
            field1_weight: Weight for the country field
            field2_weight: Weight for the province/city/district field
            field3_weight: Weight for the detailed address field
            field1_threshold: Similarity threshold for the first segment (default 0.7)
            field2_threshold: Similarity threshold for the second segment (default 0.5)
            text2vec_server_url: text2vec model service endpoint (multilingual model, port 5001)
        """
        self.field1_weight = field1_weight
        self.field2_weight = field2_weight
        self.field3_weight = field3_weight
        self.field1_threshold = field1_threshold
        self.field2_threshold = field2_threshold
        self.text2vec_server_url = text2vec_server_url.rstrip('/')
        self.split_new = split_new
        
        self.current_gpu = self._get_current_gpu()
        
        total_weight = field1_weight + field2_weight + field3_weight
        if abs(total_weight - 1.0) > 1e-6:
            print(f"[TripleFieldAccuracy2|GPU{self.current_gpu}] Warning: Weights do not sum to 1, normalizing. Original: {field1_weight}, {field2_weight}, {field3_weight}")
            self.field1_weight = field1_weight / total_weight
            self.field2_weight = field2_weight / total_weight
            self.field3_weight = field3_weight / total_weight
        
        try:
            response = requests.get(f"{self.text2vec_server_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"[TripleFieldAccuracy2|GPU{self.current_gpu}] bert service connected: {self.text2vec_server_url}")
            else:
                print(f"[TripleFieldAccuracy2|GPU{self.current_gpu}] bert service unavailable, status code: {response.status_code}")
        except Exception as e:
            print(f"[TripleFieldAccuracy2|GPU{self.current_gpu}] Unable to connect to bert service: {e}")
    
    
    def _get_current_gpu(self):
        """
        Returns the GPU id used by the current process (0~7) in Swift RLHF (colocate mode) multi-GPU training.
        """
        import os
        import torch
        import torch.distributed as dist

        local_rank = os.environ.get("LOCAL_RANK", None)
        if local_rank is not None:
            return int(local_rank)

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            num_gpus = torch.cuda.device_count()
            return rank % num_gpus

        if torch.cuda.is_available():
            return torch.cuda.current_device()

        return -1


    def _gpu_based_delay(self):
        """
        Simple delay based on GPU number.
        GPU 0: no delay
        GPU 1: delay 0.1s
        GPU 2: delay 0.2s
        GPU 3: delay 0.3s
        ...
        """
        import time
        
        gpu_id = self.current_gpu
        if gpu_id > 0:
            delay = gpu_id * 0.1
            time.sleep(delay)
    
    def _extract_answer_content(self, completion: str) -> str:
        """
        Extract answer content within <answer> tags from model output.
        
        Args:
            completion: Full text produced by the model
            
        Returns:
            Extracted answer content
        """
        if completion is None:
            return ""
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.S | re.I)
        return (m.group(1) if m else completion).strip()
    
    def _split_three_fields(self, text: str) -> tuple:
        """
        Split text into three segments.
        
        Args:
            text: Text to split
            
        Returns:
            (field1, field2, field3) tuple
        """
        if not text:
            return "", "", ""
        
        if self.split_new:
            parts = re.split(r'[;；]', text)
        else:
            parts = re.split(r'[,，]', text)
        
        parts = [part.strip() for part in parts if part.strip()]
        
        while len(parts) < 3:
            parts.append("")
        
        if len(parts) > 3:
            if self.split_new:
                parts = parts[:2] + [';'.join(parts[2:])]
            else:
                parts = parts[:2] + [','.join(parts[2:])]
        
        return parts[0], parts[1], parts[2]
    
    def _batch_cosine_similarity_text2vec(self, text_pairs: List[tuple], retry: int = 0) -> List[float]:
        """
        Batch call text2vec service to compute cosine similarity.
        
        Args:
            text_pairs: List of text pairs [(text1, text2), (text3, text4), ...]
            retry: Retry count
            
        Returns:
            List of similarities [0.95, 0.87, ...]
        """
        gpu_id = self.current_gpu
        
        if not text_pairs:
            return []
        
        try:
            pairs_data = [[text1, text2] for text1, text2 in text_pairs]
            
            response = requests.post(
                f"{self.text2vec_server_url}/batch_similarity",
                json={"pairs": pairs_data},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    similarities = result.get('similarities', [])
                    return similarities
                else:
                    error_msg = result.get('message', 'Unknown error')
                    print(f"[TripleFieldAccuracy2|GPU{gpu_id}] Service returned error: {error_msg}")
                    return [0.0] * len(text_pairs)
            else:
                print(f"[TripleFieldAccuracy2|GPU{gpu_id}] HTTP error: {response.status_code}")
                return [0.0] * len(text_pairs)
            
        except requests.exceptions.Timeout:
            if retry < 5:
                print(f"[TripleFieldAccuracy2|GPU{gpu_id}] Batch request timeout, retrying {retry+1}")
                import time
                time.sleep(0.1 * self.current_gpu)  
                return self._batch_cosine_similarity_text2vec(text_pairs, retry+1)
            print(f"[TripleFieldAccuracy2|GPU{gpu_id}] Timeout after {retry} retries, returning 0 score")
            return [0.0] * len(text_pairs)
        except Exception as e:
            print(f"[TripleFieldAccuracy2|GPU{gpu_id}] Batch cosine similarity failed: {e}")
            return [0.0] * len(text_pairs)
    
    def __call__(self, completions: List[str], solution: Union[str, List[str]], **kwargs) -> List[float]:
        """
        Compute triple field accuracy reward (cosine similarity and threshold for all three segments).
        
        Args:
            completions: List of completions generated by the model
            solution: List or single string of ground truths
            **kwargs: Other arguments
            
        Returns:
            List[float]: Rewards for each sample
        """
        gpu_id = self.current_gpu
        
        self._gpu_based_delay()
        
        if isinstance(solution, str):
            solutions = [solution] * len(completions)
        elif isinstance(solution, list):
            solutions = solution
        else:
            print(f"[TripleFieldAccuracy2|GPU{gpu_id}] Unsupported solution type: {type(solution)}, return all 0")
            return [0.0] * len(completions)
        
        if len(solutions) == 1 and len(completions) > 1:
            solutions = solutions * len(completions)
        elif len(solutions) != len(completions):
            print(f"[TripleFieldAccuracy2|GPU{gpu_id}] completions({len(completions)}) and solutions({len(solutions)}) length mismatch, return all 0")
            return [0.0] * len(completions)
        
        samples_data = []
        field1_pairs = []
        field1_pair_to_idx = []
        
        for idx, (completion, sol) in enumerate(zip(completions, solutions)):
            try:
                pred_answer = self._extract_answer_content(completion)
                gt_answer = self._extract_answer_content(str(sol))
                
                if not pred_answer or not gt_answer:
                    samples_data.append({
                        'field1_score': 0.0,
                        'field2_score': 0.0,
                        'field3_score': 0.0,
                        'valid': False
                    })
                    continue
                
                pred_field1, pred_field2, pred_field3 = self._split_three_fields(pred_answer)
                gt_field1, gt_field2, gt_field3 = self._split_three_fields(gt_answer)
                
                if pred_field1 and gt_field1:
                    field1_pairs.append((pred_field1, gt_field1))
                    field1_pair_to_idx.append(idx)
                    
                    samples_data.append({
                        'pred_field1': pred_field1,
                        'gt_field1': gt_field1,
                        'pred_field2': pred_field2,
                        'gt_field2': gt_field2,
                        'pred_field3': pred_field3,
                        'gt_field3': gt_field3,
                        'field1_score': 0.0,
                        'field2_score': 0.0,
                        'field3_score': 0.0,
                        'valid': True
                    })
                else:
                    samples_data.append({
                        'field1_score': 0.0,
                        'field2_score': 0.0,
                        'field3_score': 0.0,
                        'valid': False
                    })
                    
            except Exception as e:
                print(f"[TripleFieldAccuracy2|GPU{gpu_id}] Error processing sample {idx}: {e}")
                samples_data.append({
                    'field1_score': 0.0,
                    'field2_score': 0.0,
                    'field3_score': 0.0,
                    'valid': False
                })
        
        if field1_pairs:
            print(f"[TripleFieldAccuracy2|GPU{gpu_id}] Batch compute segment 1 similarity: {len(field1_pairs)} pairs")
            field1_similarities = self._batch_cosine_similarity_text2vec(field1_pairs)
            
            for pair_idx, sample_idx in enumerate(field1_pair_to_idx):
                if pair_idx < len(field1_similarities):
                    samples_data[sample_idx]['field1_score'] = field1_similarities[pair_idx]
        
        field2_pairs = []
        field2_pair_to_idx = []
        
        for idx, sample in enumerate(samples_data):
            if sample.get('valid') and sample['field1_score'] >= self.field1_threshold:
                if sample.get('pred_field2') and sample.get('gt_field2'):
                    field2_pairs.append((sample['pred_field2'], sample['gt_field2']))
                    field2_pair_to_idx.append(idx)
        
        if field2_pairs:
            print(f"[TripleFieldAccuracy2|GPU{gpu_id}] Batch compute segment 2 similarity: {len(field2_pairs)} pairs (filtered after segment 1)")
            field2_similarities = self._batch_cosine_similarity_text2vec(field2_pairs)
            
            for pair_idx, sample_idx in enumerate(field2_pair_to_idx):
                if pair_idx < len(field2_similarities):
                    samples_data[sample_idx]['field2_score'] = field2_similarities[pair_idx]
        
        field3_pairs = []
        field3_pair_to_idx = []
        
        for idx, sample in enumerate(samples_data):
            if sample.get('valid') and sample['field2_score'] >= self.field2_threshold:
                if sample.get('pred_field3') and sample.get('gt_field3'):
                    field3_pairs.append((sample['pred_field3'], sample['gt_field3']))
                    field3_pair_to_idx.append(idx)
        
        if field3_pairs:
            print(f"[TripleFieldAccuracy2|GPU{gpu_id}] Batch compute segment 3 similarity: {len(field3_pairs)} pairs (filtered after segment 2)")
            field3_similarities = self._batch_cosine_similarity_text2vec(field3_pairs)
            
            for pair_idx, sample_idx in enumerate(field3_pair_to_idx):
                if pair_idx < len(field3_similarities):
                    samples_data[sample_idx]['field3_score'] = field3_similarities[pair_idx]
        
        rewards = []
        for idx, sample in enumerate(samples_data):
            try:
                if not sample.get('valid'):
                    rewards.append(0.0)
                    continue
                
                field1_score = sample['field1_score']
                field2_score = sample['field2_score']
                field3_score = sample['field3_score']
                
                final_reward = (self.field1_weight * field1_score + 
                              self.field2_weight * field2_score + 
                              self.field3_weight * field3_score)
                
                rewards.append(float(final_reward))
                
            except Exception as e:
                print(f"[TripleFieldAccuracy2|GPU{gpu_id}] Error computing final score for sample {idx}: {e}")
                rewards.append(0.0)
        

        return rewards



orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'geo_format': GeoAnswerFormat,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'geoscore_accuracy':GeoScoreAccuracy,
    'triple_accuracy':TripleFieldAccuracy,
    'think_answer':ThinkAnswerConsistency
}
