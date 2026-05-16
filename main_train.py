import os
import subprocess

def run_command(command):
    print(f"\n{'='*50}\nExecuting: {command}\n{'='*50}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        exit(result.returncode)

if __name__ == "__main__":
    print("Starting Automated End-to-End Training Pipeline...")
    
    # Run Phase 2: Data Pipeline
    run_command("uv run python src/data_pipeline/run_pipeline.py")
    
    # Run Phase 3: Recall Model
    run_command("uv run python src/recall_model/run_recall.py")
    
    # Run Phase 4: Ranking Model
    run_command("uv run python src/ranking_model/run_ranking.py")
    
    print("\nAll Training Phases Completed Successfully! You can now start the server with `uv run python main_serve.py`")
