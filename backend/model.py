"""
Gap2Growth ML Backend — model.py
Training pipeline using scikit-learn on real resume data.
"""
import os, re, json, joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Comprehensive skill dictionary ─────────────────────────────────────────
SKILL_GROUPS = {
    "programming": ["python","java","javascript","typescript","c++","c#","go","rust","scala","kotlin","swift","r","matlab","perl","ruby","php","bash","shell"],
    "ml_ai": ["machine learning","deep learning","neural network","nlp","computer vision","reinforcement learning","tensorflow","pytorch","keras","scikit-learn","xgboost","lightgbm","huggingface","transformers","bert","gpt","llm","diffusion","gan"],
    "data": ["pandas","numpy","scipy","matplotlib","seaborn","plotly","tableau","power bi","sql","nosql","spark","hadoop","hive","kafka","airflow","dbt","etl","data pipeline","data warehouse","bigquery","redshift","snowflake"],
    "cloud": ["aws","azure","gcp","google cloud","lambda","ec2","s3","docker","kubernetes","terraform","ansible","jenkins","ci/cd","devops","mlops","sagemaker","vertex ai","azure ml"],
    "web": ["react","vue","angular","node.js","express","django","flask","fastapi","rest api","graphql","html","css","tailwind","next.js","typescript","webpack","vite"],
    "database": ["mysql","postgresql","mongodb","redis","elasticsearch","cassandra","dynamodb","sqlite","oracle","firebase","supabase"],
    "security": ["cybersecurity","penetration testing","ethical hacking","cryptography","owasp","soc","siem","network security","iam","oauth","jwt"],
    "mobile": ["android","ios","react native","flutter","swift","kotlin","xcode","android studio"],
    "soft": ["agile","scrum","kanban","leadership","communication","teamwork","problem solving","critical thinking","project management","jira","confluence"],
    "data_science_tools": ["jupyter","colab","vs code","git","github","gitlab","docker","linux","bash","api","rest","json","xml"],
}

ALL_SKILLS = []
for group in SKILL_GROUPS.values():
    ALL_SKILLS.extend(group)
ALL_SKILLS = list(set(ALL_SKILLS))

# ─── Job role definitions with required skill sets ───────────────────────────
JOB_ROLES = {
    "ML Engineer": {
        "required": ["python","tensorflow","pytorch","scikit-learn","machine learning","deep learning","docker","kubernetes","sql","git","mlops","linux"],
        "base_salary": 1400000, "growth": 34, "weight": 1.4,
    },
    "Data Scientist": {
        "required": ["python","r","pandas","numpy","scikit-learn","machine learning","sql","tableau","statistics","jupyter","matplotlib","deep learning"],
        "base_salary": 1200000, "growth": 28, "weight": 1.2,
    },
    "Full Stack Developer": {
        "required": ["javascript","react","node.js","html","css","sql","git","rest api","typescript","docker"],
        "base_salary": 1100000, "growth": 25, "weight": 1.1,
    },
    "Backend Developer": {
        "required": ["python","java","go","sql","postgresql","redis","docker","rest api","microservices","git","linux"],
        "base_salary": 1000000, "growth": 22, "weight": 1.0,
    },
    "Frontend Developer": {
        "required": ["javascript","typescript","react","html","css","git","webpack","rest api","testing"],
        "base_salary": 900000, "growth": 20, "weight": 0.9,
    },
    "DevOps Engineer": {
        "required": ["kubernetes","docker","terraform","aws","ci/cd","linux","bash","ansible","jenkins","monitoring"],
        "base_salary": 1300000, "growth": 31, "weight": 1.3,
    },
    "Cloud Architect": {
        "required": ["aws","azure","gcp","kubernetes","terraform","networking","security","microservices","docker","python"],
        "base_salary": 1800000, "growth": 38, "weight": 1.6,
    },
    "Data Engineer": {
        "required": ["python","sql","spark","kafka","airflow","dbt","bigquery","etl","docker","data pipeline"],
        "base_salary": 1200000, "growth": 30, "weight": 1.2,
    },
    "Cybersecurity Engineer": {
        "required": ["cybersecurity","python","linux","networking","cryptography","ethical hacking","siem","owasp"],
        "base_salary": 1400000, "growth": 32, "weight": 1.25,
    },
    "Product Manager": {
        "required": ["agile","scrum","product roadmap","jira","communication","analytics","user research","sql","leadership"],
        "base_salary": 1600000, "growth": 20, "weight": 1.2,
    },
    "iOS Developer": {
        "required": ["swift","xcode","ios","objective-c","git","rest api","testing","ui/ux"],
        "base_salary": 1200000, "growth": 18, "weight": 1.1,
    },
    "Android Developer": {
        "required": ["kotlin","java","android","android studio","rest api","git","testing"],
        "base_salary": 1100000, "growth": 18, "weight": 1.0,
    },
}

def extract_skills(text: str) -> list[str]:
    """Extract skills from resume text using keyword matching + fuzzy normalization."""
    text_lower = text.lower()
    found = []
    for skill in ALL_SKILLS:
        # Use word boundary matching for short skills, substring for phrases
        pattern = r'\b' + re.escape(skill) + r'\b' if len(skill.split()) == 1 else re.escape(skill)
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(set(found))

def generate_synthetic_training_data(n=800):
    """Generate realistic synthetic resume texts for training when no CSV is available."""
    np.random.seed(42)
    data = []
    roles = list(JOB_ROLES.keys())
    for _ in range(n):
        role = np.random.choice(roles)
        job = JOB_ROLES[role]
        required = job["required"]
        # Sample most required skills + some extra noise skills
        n_have = np.random.randint(max(1, len(required) - 5), len(required) + 1)
        skills = list(np.random.choice(required, min(n_have, len(required)), replace=False))
        noise = np.random.choice(ALL_SKILLS, np.random.randint(2, 8), replace=False)
        skills.extend(noise)
        text = " ".join(skills) + " " + " ".join([s for s in np.random.choice(required, min(3, len(required)), replace=False)])
        data.append({"text": text, "category": role})
    return pd.DataFrame(data)

def load_training_data():
    """Load Kaggle resume CSV if present, else use synthetic data."""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "resumes.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Normalize column names (Kaggle dataset uses 'Resume_str' and 'Category')
            col_map = {}
            for c in df.columns:
                if "resume" in c.lower() or "text" in c.lower():
                    col_map[c] = "text"
                if "category" in c.lower() or "label" in c.lower() or "role" in c.lower():
                    col_map[c] = "category"
            df = df.rename(columns=col_map)
            if "text" in df.columns and "category" in df.columns:
                # Map Kaggle categories to our roles where possible
                category_map = {
                    "DATA SCIENCE": "Data Scientist",
                    "MACHINE LEARNING": "ML Engineer",
                    "WEB DESIGNING": "Frontend Developer",
                    "JAVA DEVELOPER": "Backend Developer",
                    "PYTHON DEVELOPER": "Backend Developer",
                    "DEVOPS ENGINEER": "DevOps Engineer",
                    "CLOUD": "Cloud Architect",
                    "ANDROID": "Android Developer",
                    "IOS": "iOS Developer",
                    "DATA ENGINEER": "Data Engineer",
                    "CYBERSECURITY": "Cybersecurity Engineer",
                }
                df["category"] = df["category"].str.upper().map(category_map).fillna("Full Stack Developer")
                print(f"[Model] Loaded {len(df)} resumes from CSV")
                return df[["text", "category"]].dropna()
        except Exception as e:
            print(f"[Model] CSV load error: {e}, using synthetic data")

    print("[Model] Generating synthetic training data...")
    return generate_synthetic_training_data(1000)

class ResumeAnalysisModel:
    def __init__(self):
        self.classifier = None       # TF-IDF + LogReg for role classification
        self.vectorizer = None       # shared TF-IDF vectorizer
        self.salary_model = None     # GBR for salary prediction
        self.label_enc = None
        self.is_trained = False

    def train(self):
        print("[Model] Starting training...")
        df = load_training_data()

        # ── Train role classifier ──────────────────────────────────────────
        self.vectorizer = TfidfVectorizer(
            max_features=3000, ngram_range=(1, 2),
            sublinear_tf=True, stop_words="english",
        )
        X = self.vectorizer.fit_transform(df["text"])
        self.label_enc = LabelEncoder()
        y = self.label_enc.fit_transform(df["category"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        self.classifier = LogisticRegression(
            max_iter=1000, C=2.0, multi_class="multinomial", random_state=42
        )
        self.classifier.fit(X_train, y_train)
        acc = self.classifier.score(X_test, y_test)
        print(f"[Model] Role classifier accuracy: {acc:.2%}")

        # ── Train salary predictor ─────────────────────────────────────────
        sal_X, sal_y = [], []
        for _, row in df.iterrows():
            role = row["category"]
            if role in JOB_ROLES:
                skills = extract_skills(row["text"])
                required = JOB_ROLES[role]["required"]
                match = len(set(skills) & set(required)) / max(len(required), 1)
                base = JOB_ROLES[role]["base_salary"]
                sal_X.append([match, len(skills), JOB_ROLES[role].get("weight", 1.0)])
                sal_y.append(base * (0.7 + 0.6 * match + np.random.normal(0, 0.05)))

        if sal_X:
            self.salary_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )
            self.salary_model.fit(np.array(sal_X), np.array(sal_y))
            print("[Model] Salary model trained.")

        self.is_trained = True
        self._save()
        print("[Model] Training complete!")

    def _save(self):
        joblib.dump(self.vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
        joblib.dump(self.classifier, os.path.join(MODEL_DIR, "classifier.pkl"))
        joblib.dump(self.label_enc, os.path.join(MODEL_DIR, "label_enc.pkl"))
        if self.salary_model:
            joblib.dump(self.salary_model, os.path.join(MODEL_DIR, "salary_model.pkl"))
        print("[Model] Models saved to disk.")

    def load(self) -> bool:
        try:
            self.vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
            self.classifier = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))
            self.label_enc = joblib.load(os.path.join(MODEL_DIR, "label_enc.pkl"))
            sal_path = os.path.join(MODEL_DIR, "salary_model.pkl")
            if os.path.exists(sal_path):
                self.salary_model = joblib.load(sal_path)
            self.is_trained = True
            print("[Model] Loaded pre-trained models from disk.")
            return True
        except Exception as e:
            print(f"[Model] Could not load saved models: {e}")
            return False

    def analyze(self, resume_text: str, job_role: str, job_desc: str = "") -> dict:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        combined_text = resume_text + " " + job_desc
        found_skills = extract_skills(combined_text)

        # Use jobRole from frontend if valid, else classify
        if job_role not in JOB_ROLES:
            vec = self.vectorizer.transform([resume_text])
            pred_idx = self.classifier.predict(vec)[0]
            job_role = self.label_enc.inverse_transform([pred_idx])[0]

        job = JOB_ROLES.get(job_role, JOB_ROLES["Full Stack Developer"])
        required = job["required"]

        matched = [s for s in found_skills if s in required]
        missing = [s for s in required if s not in found_skills]

        skill_score = round(len(matched) / max(len(required), 1) * 100)

        # Cosine similarity between resume vector and job description
        if job_desc.strip():
            vecs = self.vectorizer.transform([resume_text, job_desc])
            sim = cosine_similarity(vecs[0], vecs[1])[0][0]
            skill_score = round((skill_score * 0.7 + float(sim) * 100 * 0.3))

        skill_score = min(skill_score, 100)

        # Confidence from classifier probability
        vec = self.vectorizer.transform([resume_text])
        proba = self.classifier.predict_proba(vec)[0]
        labels = self.label_enc.classes_
        role_idx = list(labels).index(job_role) if job_role in labels else np.argmax(proba)
        confidence = round(float(proba[role_idx]), 3)

        # Skill DNA score (weighted across skill groups)
        dna_score = self._compute_dna_score(found_skills, job_role)

        # Salary prediction
        match_ratio = len(matched) / max(len(required), 1)
        weight = job.get("weight", 1.0)
        if self.salary_model:
            pred_sal = self.salary_model.predict([[match_ratio, len(found_skills), weight]])[0]
            current_salary = max(300000, int(round(pred_sal * 0.75 / 10000) * 10000))
            potential_salary = max(current_salary, int(round(pred_sal / 10000) * 10000))
        else:
            current_salary = int(job["base_salary"] * (0.5 + 0.3 * match_ratio))
            potential_salary = int(job["base_salary"] * (0.9 + 0.3 * match_ratio))

        interview_risk = max(10, min(90, round(100 - skill_score * 0.6 - confidence * 20)))
        peer_percentile = min(95, max(10, round(skill_score * 0.8 + confidence * 20)))

        return {
            "found": found_skills,
            "missing": missing,
            "matched": matched,
            "skillScore": skill_score,
            "dnaScore": dna_score,
            "interviewRisk": interview_risk,
            "currentSalary": current_salary,
            "potentialSalary": potential_salary,
            "peerPercentile": peer_percentile,
            "jobRole": job_role,
            "confidence": confidence,
            "classifiedRole": job_role,
            "job": {
                "required": required,
                "baseSalary": job["base_salary"],
                "growth": job["growth"],
            },
        }

    def _compute_dna_score(self, found_skills: list, job_role: str) -> int:
        """Score across skill groups weighted by role relevance.
        Uses expected_coverage (4 skills per group) normalization instead of 
        total group size, which was causing artificially low scores.
        """
        role_weights = {
            "ML Engineer":    {"ml_ai": 2.5, "programming": 1.5, "cloud": 1.0, "data": 1.2},
            "Data Scientist": {"ml_ai": 2.0, "data": 2.0, "programming": 1.5, "data_science_tools": 1.0},
            "Full Stack Developer": {"web": 2.5, "database": 1.5, "programming": 1.5},
            "Backend Developer": {"programming": 2.0, "database": 1.5, "cloud": 1.0},
            "Frontend Developer": {"web": 3.0, "programming": 1.0},
            "DevOps Engineer": {"cloud": 3.0, "programming": 1.0, "data_science_tools": 0.8},
            "Cloud Architect": {"cloud": 3.5, "programming": 1.0, "database": 0.8},
            "Data Engineer":  {"data": 2.5, "programming": 1.5, "cloud": 1.0},
            "Cybersecurity Engineer": {"security": 3.0, "programming": 1.0, "cloud": 0.8},
            "iOS Developer":  {"mobile": 3.0, "programming": 1.5},
            "Android Developer": {"mobile": 3.0, "programming": 1.5},
            "Product Manager": {"soft": 2.5, "data": 1.0, "data_science_tools": 0.8},
        }
        weights = role_weights.get(job_role, {"programming": 1.5, "data_science_tools": 1.0})
        
        # Only score groups relevant to this role (weight > 0)
        EXPECTED_PER_GROUP = 4  # expect ~4 skills per group in a good resume
        
        total_w, weighted_score = 0, 0
        for group, skills in SKILL_GROUPS.items():
            found_in_group = len([s for s in found_skills if s in skills])
            # Normalize: how many expected skills does the candidate have?
            coverage = min(found_in_group / EXPECTED_PER_GROUP, 1.0)
            w = weights.get(group, 0.3)  # unrelated groups have low weight
            weighted_score += coverage * w * 100
            total_w += w
        
        # Normalize by sum of ALL weights (incl. low-weight groups)
        raw = weighted_score / max(total_w, 1)
        return min(100, max(0, round(raw)))


# Singleton
_model = ResumeAnalysisModel()

def get_model() -> ResumeAnalysisModel:
    if not _model.is_trained:
        if not _model.load():
            _model.train()
    return _model
