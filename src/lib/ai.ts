import { GoogleGenerativeAI } from "@google/generative-ai";
import { ClassificationResult, ComplaintCategory, DEPARTMENT_MAP } from "./types";

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

const CLASSIFICATION_PROMPT = `You are a civic complaint classification AI for an Indian city municipal system.

Analyze the following citizen complaint and return a JSON object with these exact fields:
- "category": one of: POTHOLE, STREETLIGHT, WATER_LEAK, GARBAGE, DRAINAGE, ROAD_DAMAGE, ILLEGAL_CONSTRUCTION, NOISE_POLLUTION, OTHER
- "confidence": a number between 0.0 and 1.0 indicating how confident you are
- "urgencyLevel": one of: CRITICAL, HIGH, MEDIUM, LOW
- "affectedAreaSize": one of: SMALL, MEDIUM, LARGE
- "estimatedPeopleAffected": estimated number of people affected (integer)
- "summary": a one-sentence summary of the issue and its severity

Guidelines for urgency:
- CRITICAL: immediate danger to life, major infrastructure failure, flooding, collapse
- HIGH: safety hazard, broken for weeks, affecting many people, worsening
- MEDIUM: ongoing issue, inconvenience, needs repair within days
- LOW: minor cosmetic issue, suggestion, request

Return ONLY the JSON object, no markdown, no extra text.`;

export async function classifyComplaint(title: string, description: string): Promise<ClassificationResult> {
    try {
        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-lite" });

        const result = await model.generateContent([
            CLASSIFICATION_PROMPT,
            `\nComplaint Title: ${title}\nComplaint Description: ${description}`,
        ]);

        const text = result.response.text().trim();

        // Strip markdown code fences if present
        const jsonStr = text.replace(/^```json?\n?/i, "").replace(/\n?```$/i, "").trim();
        const parsed = JSON.parse(jsonStr);

        // Validate and sanitize
        const validCategories: ComplaintCategory[] = [
            "POTHOLE", "STREETLIGHT", "WATER_LEAK", "GARBAGE", "DRAINAGE",
            "ROAD_DAMAGE", "ILLEGAL_CONSTRUCTION", "NOISE_POLLUTION", "OTHER",
        ];
        const validUrgency = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
        const validArea = ["SMALL", "MEDIUM", "LARGE"];

        return {
            category: validCategories.includes(parsed.category) ? parsed.category : "OTHER",
            confidence: Math.min(1, Math.max(0, Number(parsed.confidence) || 0.7)),
            urgencyLevel: validUrgency.includes(parsed.urgencyLevel) ? parsed.urgencyLevel : "MEDIUM",
            affectedAreaSize: validArea.includes(parsed.affectedAreaSize) ? parsed.affectedAreaSize : "MEDIUM",
            estimatedPeopleAffected: Math.max(1, Math.round(Number(parsed.estimatedPeopleAffected) || 50)),
            summary: parsed.summary || `Detected civic issue: ${title}`,
        };
    } catch (error) {
        console.error("Gemini classification failed, using fallback:", error);
        return fallbackClassify(title, description);
    }
}

// ===== KEYWORD FALLBACK (used if Gemini API fails) =====
function fallbackClassify(title: string, description: string): ClassificationResult {
    const text = `${title} ${description}`.toLowerCase();

    const CATEGORY_KEYWORDS: Record<ComplaintCategory, string[]> = {
        POTHOLE: ["pothole", "pot hole", "hole in road", "road hole", "crater"],
        STREETLIGHT: ["streetlight", "street light", "lamp", "light not working", "dark street", "no light", "bulb"],
        WATER_LEAK: ["water leak", "pipe burst", "broken pipe", "water flowing", "leaking pipe", "flooding"],
        GARBAGE: ["garbage", "trash", "waste", "dustbin", "rubbish", "dump", "litter", "foul smell"],
        DRAINAGE: ["drain", "drainage", "sewer", "gutter", "clogged drain", "blocked drain", "manhole"],
        ROAD_DAMAGE: ["road damage", "broken road", "crack", "road crack", "road repair", "damaged road"],
        ILLEGAL_CONSTRUCTION: ["illegal construction", "unauthorized", "encroachment"],
        NOISE_POLLUTION: ["noise", "loud", "music", "honking", "disturbance"],
        OTHER: [],
    };

    let bestCategory: ComplaintCategory = "OTHER";
    let bestScore = 0;
    for (const [category, keywords] of Object.entries(CATEGORY_KEYWORDS)) {
        let score = 0;
        for (const keyword of keywords) {
            if (text.includes(keyword)) score += keyword.split(" ").length;
        }
        if (score > bestScore) {
            bestScore = score;
            bestCategory = category as ComplaintCategory;
        }
    }

    const urgencyMap: Record<string, string[]> = {
        CRITICAL: ["urgent", "emergency", "dangerous", "accident", "burst", "flood", "collapse"],
        HIGH: ["serious", "broken", "weeks", "safety", "unsafe", "multiple"],
        MEDIUM: ["days", "problem", "issue", "complaint", "needs repair"],
        LOW: ["minor", "small", "cosmetic", "suggestion"],
    };

    let urgencyLevel: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" = "MEDIUM";
    let urgencyScore = 0;
    for (const [level, keywords] of Object.entries(urgencyMap)) {
        let score = 0;
        for (const kw of keywords) { if (text.includes(kw)) score++; }
        if (score > urgencyScore) {
            urgencyScore = score;
            urgencyLevel = level as "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
        }
    }

    const peopleMap = { SMALL: 10, MEDIUM: 50, LARGE: 200 };
    const affectedAreaSize = "MEDIUM" as const;

    return {
        category: bestCategory,
        confidence: Math.min(0.95, 0.6 + bestScore * 0.1),
        urgencyLevel,
        affectedAreaSize,
        estimatedPeopleAffected: peopleMap[affectedAreaSize],
        summary: `Detected ${bestCategory.replace("_", " ").toLowerCase()} issue with ${urgencyLevel.toLowerCase()} urgency.`,
    };
}

export function calculatePriorityScore(
    urgencyLevel: string,
    affectedAreaSize: string,
    estimatedPeopleAffected: number,
    upvoteCount: number = 0
): { score: number; level: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" } {
    const urgencyMap: Record<string, number> = { CRITICAL: 10, HIGH: 7, MEDIUM: 5, LOW: 2 };
    const areaMap: Record<string, number> = { LARGE: 10, MEDIUM: 6, SMALL: 3 };

    const urgencyValue = urgencyMap[urgencyLevel] || 5;
    const impactValue = Math.min(10, estimatedPeopleAffected / 20);
    const spreadValue = areaMap[affectedAreaSize] || 5;
    const votesValue = Math.min(10, upvoteCount / 5);

    const score =
        urgencyValue * 0.4 + impactValue * 0.3 + spreadValue * 0.2 + votesValue * 0.1;

    let level: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
    if (score >= 8) level = "CRITICAL";
    else if (score >= 6) level = "HIGH";
    else if (score >= 4) level = "MEDIUM";
    else level = "LOW";

    return { score: Math.round(score * 10) / 10, level };
}

export function getEstimatedResolution(priority: string): string {
    const now = new Date();
    const daysMap: Record<string, number> = {
        CRITICAL: 1,
        HIGH: 3,
        MEDIUM: 7,
        LOW: 14,
    };
    const days = daysMap[priority] || 7;
    now.setDate(now.getDate() + days);
    return now.toISOString();
}

export function getDepartment(category: ComplaintCategory): string {
    return DEPARTMENT_MAP[category] || "General Administration";
}
