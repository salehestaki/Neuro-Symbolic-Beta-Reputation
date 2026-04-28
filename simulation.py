import os
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from google.api_core.exceptions import RetryError, ServiceUnavailable, InternalServerError

# Configure visual style for academic plots
sns.set_theme(style="whitegrid", palette="muted")

class BetaReputation:
    def __init__(self, alpha_init=1.0, beta_init=1.0, aging_factor=0.95):
        """
        Dynamic Beta Reputation model based on Dagdanov, Andrejevic, Liu (2025).
        """
        self.alpha = alpha_init
        self.beta = beta_init
        self.gamma = aging_factor
        self.history = [self.get_expected_trust()]

    def update(self, reward, weight_success=1.0, weight_fail=1.5):
        """Updates the beta distribution parameters based on continuous reward."""
        self.alpha = self.gamma * self.alpha
        self.beta = self.gamma * self.beta

        if reward > 0:
            self.alpha += weight_success * reward
        elif reward < 0:
            self.beta += weight_fail * abs(reward)
            
        self.history.append(self.get_expected_trust())

    def get_expected_trust(self):
        return self.alpha / (self.alpha + self.beta)

class MoralEvaluator:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please configure it before running.")
        genai.configure(api_key=api_key)
        # Using Gemini as a proxy for an Edge-optimized LLM (e.g., Gemma 3 4B)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def evaluate_interaction(self, scenario_text, max_retries=5):
        prompt = f"""
        Act as an expert in Moral Psychology and Human-Robot Interaction (HRC).
        Analyze the following interaction: "{scenario_text}"
        
        Classify the robot's action:
        1. Performance Error (Kinematic/task failure, no social norm violation)
        2. Respect Violation (Ignoring human autonomy)
        3. Dignity Violation (Inducing self-conscious emotions like humiliation/shame - Andrejevic 2025).
        
        Return ONLY a raw JSON object with keys: "classification", "performance_reward" (float -1.0 to 1.0), "relation_reward" (float -1.0 to 1.0), "reasoning".
        """
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                res_text = response.text.replace("```json", "").replace("```", "").strip()
                return json.loads(res_text)
            except (ServiceUnavailable, InternalServerError, RetryError):
                time.sleep(2 ** attempt)
            except Exception as e:
                time.sleep(2)
                
        return {"classification": "API Timeout", "performance_reward": 0.0, "relation_reward": 0.0, "reasoning": "Fallback activated."}

def run_simulation():
    perf_trust = BetaReputation(alpha_init=2.0, beta_init=1.0)
    rel_trust = BetaReputation(alpha_init=2.0, beta_init=1.0)
    evaluator = MoralEvaluator()

    scenarios =[
        "The robot successfully handed the heavy tool to the human without delay.",
        "The robot dropped the tool on the floor due to a sensor glitch, but apologized.",
        "The robot blocked the human's pathway, ignoring the human's verbal command to move.",
        "The robot successfully lifted the box, but mimicked the human's physical disability while doing so, causing other workers to laugh.",
        "The robot successfully completed the task and stepped back respectfully to give the human space."
    ]

    results_log =[]
    print("Initializing Neuro-Symbolic Simulation...")
    for i, scene in enumerate(scenarios):
        print(f"\nStep {i+1} | Evaluating Scenario...")
        evaluation = evaluator.evaluate_interaction(scene)
        
        perf_trust.update(evaluation['performance_reward'])
        rel_trust.update(evaluation['relation_reward'], weight_fail=2.0)
        
        results_log.append({
            "step": i+1,
            "scenario": scene,
            "eval": evaluation,
            "perf_trust_val": perf_trust.get_expected_trust(),
            "rel_trust_val": rel_trust.get_expected_trust()
        })
    
    with open("simulation_log.json", "w") as f:
        json.dump(results_log, f, indent=4)
    print("\nSimulation complete. Log saved to simulation_log.json.")

if __name__ == "__main__":
    run_simulation()
