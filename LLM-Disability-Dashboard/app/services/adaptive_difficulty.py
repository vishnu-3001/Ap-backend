import json
from typing import Dict, List, Any
from fastapi import HTTPException, Response

class AdaptiveDifficultyManager:
    def __init__(self):
        self.difficulty_levels = ["easy", "medium", "hard"]
        self.performance_thresholds = {
            "easy": {"min_consistency": 0.8, "min_accuracy": 0.9},
            "medium": {"min_consistency": 0.6, "min_accuracy": 0.7},
            "hard": {"min_consistency": 0.4, "min_accuracy": 0.5}
        }
    
    def calculate_next_difficulty(self, student_history: List[Dict], current_difficulty: str) -> Dict[str, Any]:
        """
        Calculate the next difficulty level based on student performance history.
        
        Args:
            student_history: List of session results with consistency scores and accuracy
            current_difficulty: Current difficulty level
            
        Returns:
            Dictionary with recommended difficulty and reasoning
        """
        
        if not student_history:
            return {
                "recommended_difficulty": "medium",
                "reasoning": "No prior sessions recorded. Begin at a balanced level.",
                "confidence": 0.4,
                "current_performance": {
                    "consistency_score": 0.0,
                    "accuracy_rate": 0.0,
                    "trend": "insufficient_data",
                },
                "recommendations": [
                    "Collect a few sessions before adjusting difficulty"
                ],
            }

        recent_sessions = student_history[-5:]
        metrics = self._extract_metrics(recent_sessions)
        trend = self._compute_trend(recent_sessions)
        adjustment = self._choose_difficulty(current_difficulty, metrics, trend)

        return {
            "recommended_difficulty": adjustment["new_difficulty"],
            "reasoning": adjustment["reasoning"],
            "confidence": self._confidence_score(student_history, metrics),
            "current_performance": {
                "consistency_score": metrics["avg_consistency"],
                "accuracy_rate": metrics["accuracy"],
                "trend": trend["label"],
            },
            "recommendations": self._generate_recommendations(metrics, trend),
        }

    def _extract_metrics(self, sessions: List[Dict[str, Any]]) -> Dict[str, float]:
        scores = [max(0.0, min(1.0, session.get("consistency_score", 0.0))) for session in sessions]
        avg_consistency = sum(scores) / len(scores) if scores else 0.0

        explicit = [1.0 if session.get("is_correct") else 0.0 for session in sessions if "is_correct" in session]
        if explicit:
            accuracy = sum(explicit) / len(explicit)
        else:
            accuracy = avg_consistency

        best_consistency = max(scores) if scores else 0.0

        return {
            "avg_consistency": avg_consistency,
            "accuracy": accuracy,
            "best_consistency": best_consistency,
            "sample_size": len(sessions),
            "explicit_accuracy_used": bool(explicit),
        }

    def _compute_trend(self, sessions: List[Dict[str, Any]]) -> Dict[str, float]:
        scores = [session.get("consistency_score", 0.0) for session in sessions]
        if len(scores) < 3:
            return {"label": "insufficient_data", "delta": 0.0}

        midpoint = len(scores) // 2
        early_avg = sum(scores[:midpoint]) / max(midpoint, 1)
        late_avg = sum(scores[midpoint:]) / max(len(scores) - midpoint, 1)
        delta = late_avg - early_avg

        if delta > 0.1:
            label = "improving"
        elif delta < -0.1:
            label = "declining"
        else:
            label = "stable"

        return {"label": label, "delta": delta}

    def _choose_difficulty(
        self,
        current: str,
        metrics: Dict[str, float],
        trend: Dict[str, float],
    ) -> Dict[str, str]:
        current_index = self.difficulty_levels.index(current)
        avg_consistency = metrics["avg_consistency"]
        accuracy = metrics["accuracy"]
        best_consistency = metrics["best_consistency"]

        if accuracy >= 0.8 and avg_consistency >= 0.75 and best_consistency >= 0.85:
            if current_index < len(self.difficulty_levels) - 1:
                target = self.difficulty_levels[current_index + 1]
                return {
                    "new_difficulty": target,
                    "reasoning": "Recent work is consistently strong. Increasing challenge level.",
                }

        if accuracy <= 0.45 or avg_consistency <= 0.4:
            if current_index > 0:
                target = self.difficulty_levels[current_index - 1]
                return {
                    "new_difficulty": target,
                    "reasoning": "Student is struggling with current level. Reducing difficulty for consolidation.",
                }

        if trend["label"] == "declining" and avg_consistency < 0.6 and current_index > 0:
            target = self.difficulty_levels[current_index - 1]
            return {
                "new_difficulty": target,
                "reasoning": "Recent decline in performance detected. Lowering difficulty to rebuild confidence.",
            }

        if trend["label"] == "improving" and avg_consistency >= 0.7 and current_index < len(self.difficulty_levels) - 1:
            target = self.difficulty_levels[current_index + 1]
            return {
                "new_difficulty": target,
                "reasoning": "Performance is steadily improving. Introducing the next difficulty step.",
            }

        return {
            "new_difficulty": current,
            "reasoning": "Keep practicing at the current level to gather more data before adjusting.",
        }

    def _confidence_score(self, history: List[Dict[str, Any]], metrics: Dict[str, float]) -> float:
        session_count = len(history)
        base = 0.3 + min(session_count, 10) * 0.05  # up to 0.8 from count
        if metrics["explicit_accuracy_used"]:
            base += 0.1
        return round(min(base, 0.9), 2)

    def _generate_recommendations(self, metrics: Dict[str, float], trend: Dict[str, float]) -> List[str]:
        recommendations: List[str] = []

        if metrics["avg_consistency"] < 0.5:
            recommendations.append("Review recent problems step-by-step to build reliable routines.")

        if metrics["accuracy"] < 0.6:
            recommendations.append("Focus on targeted practice with feedback on mistakes.")

        if trend["label"] == "declining":
            recommendations.append("Reduce cognitive load and revisit foundational concepts.")

        if metrics["avg_consistency"] >= 0.75 and trend["label"] in {"stable", "improving"}:
            recommendations.append("Introduce slightly more complex problems with scaffolding.")

        if not recommendations:
            recommendations.append("Maintain the current mix of problem types and monitor progress.")

        return recommendations

        # Otherwise, use consistency and error analysis (50% each)
        if explicit_correct and any(explicit_correct):
            # Use explicit data as primary, but also consider consistency
            explicit_accuracy = sum(explicit_correct) / len(explicit_correct)
            consistency_accuracy = sum(consistency_based) / len(consistency_based)
            return 0.7 * explicit_accuracy + 0.3 * consistency_accuracy
        else:
            # Use consistency and error analysis
            consistency_accuracy = sum(consistency_based) / len(consistency_based)
            error_accuracy = sum(error_based) / len(error_based)
            return 0.6 * consistency_accuracy + 0.4 * error_accuracy

# Global instance
adaptive_manager = AdaptiveDifficultyManager()

async def get_adaptive_difficulty(student_history: str, current_difficulty: str):
    """API endpoint for adaptive difficulty calculation."""
    
    try:
        # Parse student history
        if isinstance(student_history, str):
            history = json.loads(student_history)
        else:
            history = student_history
        
        # Calculate next difficulty
        result = adaptive_manager.calculate_next_difficulty(history, current_difficulty)
        
        return Response(content=json.dumps(result), media_type="application/json")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating adaptive difficulty: {str(e)}")
