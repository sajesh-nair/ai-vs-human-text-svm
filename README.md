# Hyperplane Studio: Hybrid Text Classification Engine

I built Hyperplane Studio because I was honestly tired of how rigid standard text detection tools are. Right now, most setups force you into a frustrating bottleneck: 
you either rely on cheap local rules that completely miss deep semantic context, or you end up routing every single text check to a massive cloud LLM, 
which destroys your infrastructure budget and drags down performance latency numbers instantly.
This project completely fixes that problem by pairing a fast local Support Vector Machine (SVM) with smart cloud routing.

## The Core Workflow
The architecture runs on a cost-aware dual-tier routing setup. 
First, a lightweight SVM running an RBF kernel serves as our first line of defense. 
It captures obvious human writing or blatant AI text right on the device, deflecting the cost entirely to save your cloud API token budget. 

What happens to the tricky, borderline text profiles? If an input lands too close to the decision boundary (a hyperplane distance threshold less than 0.85), 
the pipeline marks it as an edge case. Instead of letting the local engine guess, the system securely routes only these specific borderline cases 
to a heavyweight cloud model (llama-3.3-70b-versatile via Groq) for deep contextual validation.

## The Telemetry Matrix
We completely threw out the basic keyword whack-a-mole strategy. Instead, the feature engineering engine calculates deep structural fingerprints directly from the text.
It tracks the standard deviation of sentence lengths. Since large language models naturally write with a highly uniform machine pulse, 
tracking sentence rhythm variance helps the engine spot flat, robotic prose flows instantly. 
The system also calculates the distribution of unique words against the total word count to isolate organic human writing from highly optimized AI copy.

## Interactive Pipeline Tuning
The dashboard UI is built with Streamlit and streamlit-echarts, giving you direct, visual control over the mathematical boundary spaces. 
You can dynamically swap between rbf, linear, and poly types right from the sidebar. 
Moving the margin regularization slider changes the boundary stiffness, letting you watch the support vector coordinates shift live on the visualizer. 
Your target text is projected live as a pink diamond point relative to baseline data distributions, so you can see exactly where the engine makes its decision.

## Quick Start
### 1. Grab the repository
git clone https://github.com/sajesh-nair/ai-vs-human-text-svm.git
cd ai-vs-human-text-svm

### 2. Setup the virtual environment
python -m venv env
source env/bin/activate
pip install -r requirements.txt

### 3. Spin up the application
streamlit run app.py

---
Developed by Sajesh Nair
Framework trained on the AI-Generated Text Detection Benchmark Array.
