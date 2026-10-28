import React, { useState } from "react";
import { Txt, Row, Grid, Card, Badge, Progress } from "./ui/Primitives";
import { rand } from "../utils";

const QUESTIONS: Record<string, { technical: string[], behavioral: string[], situational: string[] }> = {
    "ML Engineer": {
        technical: [
            "Explain the difference between supervised and unsupervised learning with real examples.",
            "How do you handle class imbalance in a machine learning dataset?",
            "What is gradient descent and when would you use Adam optimizer instead?",
            "Describe bias-variance tradeoff and how do you detect overfitting?",
            "How would you deploy a machine learning model to production at scale?",
            "Explain transformer architecture and self-attention mechanism.",
        ],
        behavioral: [
            "Tell me about a time you improved a model's accuracy significantly.",
            "Describe a situation where you had to explain a complex ML concept to a non-technical stakeholder.",
            "How do you stay updated with the latest research papers and techniques?",
            "Tell me about a failure in a machine learning project and what you learned.",
        ],
        situational: [
            "Your model performs well in testing but poorly in production — what do you do?",
            "A stakeholder wants 95% accuracy but your best model achieves 88%. How do you respond?",
            "You have a dataset with 70% missing values in a key feature. How do you proceed?",
        ],
    },
    "default": {
        technical: [
            "Walk me through your technical stack and your strongest area.",
            "How do you approach debugging a complex issue in production?",
            "Describe your experience with version control and code review processes.",
            "How do you ensure code quality and maintainability in your projects?",
            "Explain a complex technical concept you've recently worked with.",
            "How do you approach system design for scalability?",
        ],
        behavioral: [
            "Tell me about a challenging project and how you overcame obstacles.",
            "Describe a time you had a conflict with a team member and how you resolved it.",
            "How do you prioritize tasks when facing multiple deadlines?",
            "Tell me about a time you took initiative and the outcome.",
        ],
        situational: [
            "A critical bug is found 1 hour before a major release — what do you do?",
            "Your team disagrees on a technical approach. How do you reach a decision?",
            "You're asked to finish a 2-week project in 3 days. How do you respond?",
        ],
    },
};

const TIPS: Record<string, string[]> = {
    "ML Engineer": [
        "Practice explaining ML concepts with real-world analogies — avoid jargon",
        "Always mention metrics (accuracy, F1, AUC) when discussing model performance",
        "Show Python/TensorFlow code fluency — LeetCode 'medium' level is expected",
        "Be ready to discuss tradeoffs: model complexity vs. interpretability",
        "Know MLOps basics: Docker, CI/CD pipelines, model monitoring",
    ],
    "default": [
        "Use the STAR method: Situation, Task, Action, Result",
        "Research the company's tech stack before the interview",
        "Prepare 2-3 specific examples of past achievements with metrics",
        "Ask thoughtful questions — it shows genuine interest",
        "Practice explaining your work simply and clearly",
    ],
};

function QuestionCard({ q, idx, T }: any) {
    const [open, setOpen] = useState(false);
    const [answered, setAnswered] = useState(false);

    const sampleAnswers: Record<number, string> = {
        0: "Start with a clear definition, then provide a concrete industry example. Structure: definition → distinction → real-world use case → why it matters.",
        1: "Mention techniques: oversampling (SMOTE), undersampling, class_weight parameter, precision-recall curves over accuracy.",
        2: "Gradient descent minimizes loss by iteratively adjusting weights. Adam combines momentum + RMSprop — use it for sparse gradients and large datasets.",
    };

    return (
        <div style={{ background: T.cardBg, border: `1px solid ${answered ? T.accent3 : T.border}`, borderRadius: 12, transition: "all .2s", marginBottom: 10 }}>
            <div
                style={{ display: "flex", alignItems: "flex-start", gap: 12, padding: "14px 16px", cursor: "pointer" }}
                onClick={() => setOpen(o => !o)}
            >
                <div style={{ width: 28, height: 28, borderRadius: "50%", background: answered ? T.accent3 : T.accent + "20", border: `2px solid ${answered ? T.accent3 : T.accent + "44"}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 700, color: answered ? "#fff" : T.accent, flexShrink: 0, marginTop: 1 }}>
                    {answered ? "✓" : idx + 1}
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 14, fontWeight: 600, color: T.text, lineHeight: 1.5 }}>{q}</div>
                    {!open && <div style={{ fontSize: 11, color: T.muted, marginTop: 4 }}>{open ? "Click to collapse" : "Click for hints & sample answer"}</div>}
                </div>
                <span style={{ color: T.muted, fontSize: 14 }}>{open ? "▲" : "▼"}</span>
            </div>
            {open && (
                <div style={{ padding: "0 16px 16px", borderTop: `1px solid ${T.border}`, paddingTop: 14, animation: "fadeIn .2s ease" }}>
                    <div style={{ background: T.accent + "10", border: `1px solid ${T.accent}25`, borderRadius: 8, padding: "12px 14px", marginBottom: 12 }}>
                        <div style={{ fontSize: 11, fontWeight: 700, color: T.accent, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.07em" }}>💡 Answer Framework</div>
                        <div style={{ fontSize: 13, color: T.text, lineHeight: 1.6 }}>
                            {sampleAnswers[idx] || "Use the STAR method: Situation → Task → Action → Result. Quantify your impact where possible. Keep answer to 2-3 minutes."}
                        </div>
                    </div>
                    <Row justify="flex-end">
                        <button
                            onClick={e => { e.stopPropagation(); setAnswered(a => !a); }}
                            style={{ background: answered ? T.accent3 + "20" : T.accent + "15", border: `1px solid ${answered ? T.accent3 : T.accent}44`, color: answered ? T.accent3 : T.accent, borderRadius: 8, padding: "6px 14px", cursor: "pointer", fontSize: 12, fontWeight: 600 }}>
                            {answered ? "✓ Marked Done" : "Mark as Practiced"}
                        </button>
                    </Row>
                </div>
            )}
        </div>
    );
}

export function MockInterviewTab({ T, jobRole, missing, found, skillScore }: any) {
    const [category, setCategory] = useState("technical");
    const qMap = QUESTIONS[jobRole] || QUESTIONS["default"];
    const tips = TIPS[jobRole] || TIPS["default"];
    const readiness = Math.round((100 - (missing.length / Math.max(found.length + missing.length, 1)) * 100) * 0.7 + skillScore * 0.3);

    const cats = [
        { id: "technical", label: "Technical", icon: "⚙️", count: qMap.technical.length },
        { id: "behavioral", label: "Behavioral", icon: "🧠", count: qMap.behavioral.length },
        { id: "situational", label: "Situational", icon: "🎯", count: qMap.situational.length },
    ];

    const activeQs = qMap[category as keyof typeof qMap] || [];

    return (
        <div>
            {/* Header */}
            <Row gap={14} wrap style={{ marginBottom: 22 }}>
                <div>
                    <Txt sz={22} w={800} c={T.text} mb={4}>🎤 Mock Interview Prep</Txt>
                    <Txt sz={13} c={T.muted}>AI-curated questions for <strong style={{ color: T.accent }}>{jobRole}</strong> — practice, review hints, and track your readiness</Txt>
                </div>
            </Row>

            {/* Readiness + Tips row */}
            <Grid cols="1fr 1fr" gap={16} style={{ marginBottom: 20 }}>
                {/* Readiness score */}
                <div style={{ background: `linear-gradient(135deg,${T.accent},${T.accent2})`, borderRadius: 16, padding: "20px 22px", position: "relative", overflow: "hidden", boxShadow: `0 6px 20px ${T.accent}40` }}>
                    <div style={{ position: "absolute", top: -20, right: -20, width: 100, height: 100, borderRadius: "50%", background: "rgba(255,255,255,0.1)" }} />
                    <div style={{ position: "relative", zIndex: 1 }}>
                        <div style={{ fontSize: 12, fontWeight: 700, color: "rgba(255,255,255,0.7)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.08em" }}>Interview Readiness</div>
                        <div style={{ fontSize: 44, fontWeight: 900, color: "#fff", lineHeight: 1, marginBottom: 8 }}>{readiness}%</div>
                        <div style={{ height: 6, background: "rgba(255,255,255,0.25)", borderRadius: 99, marginBottom: 8 }}>
                            <div style={{ height: "100%", width: `${readiness}%`, background: "#fff", borderRadius: 99, transition: "width .6s ease" }} />
                        </div>
                        <div style={{ fontSize: 12, color: "rgba(255,255,255,0.7)" }}>
                            {readiness >= 75 ? "🟢 Ready to interview!" : readiness >= 50 ? "🟡 Almost there — keep practicing" : "🔴 More prep needed"}
                        </div>
                    </div>
                </div>

                {/* Tips */}
                <Card T={T}>
                    <Txt sz={14} w={700} c={T.text} mb={12}>🏆 Interview Tips for {jobRole}</Txt>
                    {tips.map((tip, i) => (
                        <Row key={i} gap={8} style={{ marginBottom: 8, alignItems: "flex-start" }}>
                            <div style={{ width: 20, height: 20, borderRadius: "50%", background: T.accent + "20", color: T.accent, fontSize: 10, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>{i + 1}</div>
                            <Txt sz={12} c={T.muted} style={{ lineHeight: 1.5 }}>{tip}</Txt>
                        </Row>
                    ))}
                </Card>
            </Grid>

            {/* Category tabs */}
            <Row gap={10} style={{ marginBottom: 18 }}>
                {cats.map(c => (
                    <button key={c.id} onClick={() => setCategory(c.id)} style={{
                        padding: "9px 18px", borderRadius: 10, fontSize: 13, fontWeight: 600, cursor: "pointer", transition: "all .15s",
                        background: category === c.id ? T.accent : T.cardBg,
                        color: category === c.id ? "#fff" : T.muted,
                        border: `1px solid ${category === c.id ? T.accent : T.border}`,
                        boxShadow: category === c.id ? `0 4px 14px ${T.accent}44` : "none",
                    }}>
                        {c.icon} {c.label} <span style={{ background: category === c.id ? "rgba(255,255,255,0.2)" : T.accent + "20", color: category === c.id ? "#fff" : T.accent, borderRadius: 99, padding: "1px 7px", fontSize: 10, marginLeft: 4 }}>{c.count}</span>
                    </button>
                ))}
                <div style={{ marginLeft: "auto", fontSize: 12, color: T.muted, alignSelf: "center" }}>Click a question to see hints →</div>
            </Row>

            {/* Questions */}
            <div>
                {activeQs.map((q: string, i: number) => (
                    <QuestionCard key={`${category}-${i}`} q={q} idx={i} T={T} />
                ))}
            </div>

            {/* Missing skills alert */}
            {missing.length > 0 && (
                <Card T={T} style={{ marginTop: 20, borderColor: T.warn + "44", background: T.warn + "08" }}>
                    <Row gap={10} style={{ marginBottom: 10 }}>
                        <span style={{ fontSize: 18 }}>⚠️</span>
                        <Txt sz={14} w={700} c={T.text}>Likely Weak Points in Interview</Txt>
                    </Row>
                    <Txt sz={12} c={T.muted} mb={10}>Based on your resume, interviewers will likely probe these missing skills:</Txt>
                    <Row gap={8} wrap>
                        {missing.slice(0, 8).map((s: string) => (
                            <Badge key={s} color={T.warn}>{s} — study this!</Badge>
                        ))}
                    </Row>
                </Card>
            )}
        </div>
    );
}
