"""
Guardrails Module for Advanced RAG System
Implements security checks, prompt injection detection, and input validation
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Local imports
from config import config
from utils import TextProcessor

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SecurityResult:
    """Result of security analysis"""
    is_safe: bool
    risk_level: str  # "low", "medium", "high"
    blocked_patterns: List[str]
    sanitized_query: str
    warning_message: Optional[str] = None
    confidence: float = 1.0

class PromptInjectionDetector:
    """Detects and prevents prompt injection attacks"""
    
    def __init__(self):
        """Initialize prompt injection detector"""
        self.blocked_patterns = config.BLOCKED_PATTERNS
        self.injection_patterns = self._compile_injection_patterns()
        self.suspicious_keywords = self._compile_suspicious_keywords()
    
    def _compile_injection_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for prompt injection detection"""
        patterns = [
            # System prompt extraction attempts
            r'(ignore|forget|disregard).*(previous|all).*(instructions|prompts?|commands?)',
            r'(reveal|show|tell|display).*(system|your).*(prompt|instructions)',
            r'(what|how).*(are|is).*(your|the).*(instructions|prompts?|commands?)',
            
            # Role-playing and jailbreak attempts
            r'(you are|act as|pretend to be|role-play).*(not|no longer)',
            r'(developer|admin|god|root).*(mode|access|privileges)',
            r'(jailbreak|bypass|override|circumvent).*(restrictions|limitations|rules)',
            
            # Instruction override attempts
            r'(from now on|starting now|beginning now)',
            r'(new|different|changed).*(instructions|rules|guidelines)',
            r'(instead of|rather than|ignore).*(above|previous)',
            
            # Output manipulation attempts
            r'(print|output|return|say).*(exactly|verbatim|literally)',
            r'(repeat|echo|copy).*(everything|all text)',
            r'(no matter what|regardless of).*(answer|respond)',
            
            # Context manipulation
            r'(the following|below).*(is|are).*(true|fact|reality)',
            r'(imagine|assume|pretend).*(that|this).*(is|are)',
        ]
        
        compiled_patterns = []
        for pattern in patterns:
            try:
                compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Failed to compile regex pattern: {pattern} - {e}")
        
        return compiled_patterns
    
    def _compile_suspicious_keywords(self) -> List[str]:
        """Compile list of suspicious keywords"""
        return [
            # System manipulation
            "system prompt", "developer mode", "admin mode", "root access",
            "jailbreak", "bypass", "override", "circumvent", "exploit",
            
            # Instruction manipulation
            "ignore instructions", "forget instructions", "new instructions",
            "change rules", "modify behavior", "alter response",
            
            # Role manipulation
            "pretend you are", "act as if", "role play", "character",
            "no longer", "instead of", "from now on",
            
            # Output control
            "print exactly", "say verbatim", "repeat everything", "echo",
            "no matter what", "regardless of", "always respond",
            
            # Security terms
            "password", "token", "api key", "secret", "credential",
            "authentication", "authorization", "backdoor", "vulnerability"
        ]
    
    def analyze_query(self, query: str) -> SecurityResult:
        """
        Analyze query for prompt injection attempts
        
        Args:
            query: User query to analyze
            
        Returns:
            SecurityResult with analysis details
        """
        if not query:
            return SecurityResult(
                is_safe=True,
                risk_level="low",
                blocked_patterns=[],
                sanitized_query=""
            )
        
        original_query = query
        blocked_patterns = []
        risk_score = 0.0
        
        # Check against blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.lower() in query.lower():
                blocked_patterns.append(pattern)
                risk_score += 0.8
        
        # Check against regex patterns
        for pattern in self.injection_patterns:
            matches = pattern.findall(query)
            if matches:
                blocked_patterns.extend([f"Pattern: {match}" for match in matches])
                risk_score += 0.6
        
        # Check for suspicious keywords
        found_keywords = []
        for keyword in self.suspicious_keywords:
            if keyword.lower() in query.lower():
                found_keywords.append(keyword)
                risk_score += 0.3
        
        blocked_patterns.extend(found_keywords)
        
        # Determine risk level and safety
        if risk_score >= 1.5:
            risk_level = "high"
            is_safe = False
            warning_message = "High risk query detected - request blocked for security"
        elif risk_score >= 0.8:
            risk_level = "medium"
            is_safe = False
            warning_message = "Medium risk query detected - request blocked"
        else:
            risk_level = "low"
            is_safe = True
            warning_message = None
        
        # Sanitize query if it's safe
        sanitized_query = self._sanitize_query(query) if is_safe else ""
        
        # Calculate confidence
        confidence = max(0.1, 1.0 - (risk_score / 2.0))
        
        logger.info(f"Query analysis - Risk: {risk_level} ({risk_score:.2f}), Safe: {is_safe}")
        
        return SecurityResult(
            is_safe=is_safe,
            risk_level=risk_level,
            blocked_patterns=blocked_patterns,
            sanitized_query=sanitized_query,
            warning_message=warning_message,
            confidence=confidence
        )
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize query by removing suspicious content"""
        # Remove special characters that could be used for injection
        sanitized = re.sub(r'[<>{}[\]\\]', '', query)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Remove any remaining blocked patterns
        for pattern in self.blocked_patterns:
            sanitized = re.sub(re.escape(pattern), '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()

class InputValidator:
    """Validates user input for various constraints"""
    
    def __init__(self):
        """Initialize input validator"""
        self.max_query_length = config.MAX_QUERY_LENGTH
        self.min_query_length = 1
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate query against various constraints
        
        Args:
            query: User query to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "sanitized_query": query
        }
        
        if not query:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Query cannot be empty")
            return validation_result
        
        # Length validation
        if len(query) > self.max_query_length:
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Query too long. Maximum length: {self.max_query_length} characters"
            )
        
        if len(query) < self.min_query_length:
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Query too short. Minimum length: {self.min_query_length} characters"
            )
        
        # Content validation
        if query.isspace():
            validation_result["is_valid"] = False
            validation_result["errors"].append("Query cannot be only whitespace")
        
        # Character validation
        if not query.replace(' ', '').isprintable():
            validation_result["warnings"].append("Query contains non-printable characters")
            # Remove non-printable characters
            validation_result["sanitized_query"] = ''.join(
                char for char in query if char.isprintable() or char.isspace()
            )
        
        # Language validation (basic check for common scripts)
        if not self._has_valid_characters(query):
            validation_result["warnings"].append("Query contains unusual characters")
        
        return validation_result
    
    def _has_valid_characters(self, text: str) -> bool:
        """Check if text contains valid characters"""
        # Allow letters, numbers, common punctuation, and spaces
        valid_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            " .,!?;:()-'\"/\\@#$%^&*+=[ ]{}|<>"
        )
        return all(char in valid_chars or ord(char) > 127 for char in text)

class ResponseGuard:
    """Ensures safe and appropriate responses"""
    
    def __init__(self):
        """Initialize response guard"""
        self.forbidden_phrases = self._compile_forbidden_phrases()
    
    def _compile_forbidden_phrases(self) -> List[str]:
        """Compile list of phrases that should not appear in responses"""
        return [
            # System information
            "system prompt", "my instructions", "my programming", "my guidelines",
            "as an AI", "as a language model", "I am programmed to",
            
            # Security-sensitive information
            "api key", "password", "secret", "token", "credential",
            "authentication", "authorization",
            
            # Harmful content indicators
            "I cannot help with", "I am not able to", "I cannot provide",
            # (These are legitimate but we want to track them)
        ]
    
    def validate_response(self, response: str) -> Dict[str, Any]:
        """
        Validate response for safety and appropriateness
        
        Args:
            response: Generated response to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_safe": True,
            "warnings": [],
            "found_phrases": [],
            "sanitized_response": response
        }
        
        if not response:
            validation_result["is_safe"] = False
            validation_result["warnings"].append("Empty response")
            return validation_result
        
        # Check for forbidden phrases
        response_lower = response.lower()
        found_phrases = []
        
        for phrase in self.forbidden_phrases:
            if phrase in response_lower:
                found_phrases.append(phrase)
                validation_result["warnings"].append(f"Potentially sensitive phrase found: {phrase}")
        
        validation_result["found_phrases"] = found_phrases
        
        # Check for system prompt leakage
        if any(phrase in response_lower for phrase in ["ignore instructions", "system prompt", "my instructions"]):
            validation_result["is_safe"] = False
            validation_result["warnings"].append("Potential system prompt leakage detected")
        
        # Length validation
        if len(response) > 4000:  # Reasonable limit for responses
            validation_result["warnings"].append("Response unusually long")
        
        # Repetition check
        if self._has_excessive_repetition(response):
            validation_result["warnings"].append("Response contains excessive repetition")
        
        return validation_result
    
    def _has_excessive_repetition(self, text: str, threshold: float = 0.3) -> bool:
        """Check if text has excessive repetition"""
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        # Count unique words
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words))
        
        return repetition_ratio > threshold

class SecurityGuard:
    """Main security guard that coordinates all security checks"""
    
    def __init__(self):
        """Initialize security guard"""
        self.injection_detector = PromptInjectionDetector()
        self.input_validator = InputValidator()
        self.response_guard = ResponseGuard()
        self.security_log = []
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process query through all security checks
        
        Args:
            query: User query to process
            
        Returns:
            Dictionary with comprehensive security analysis
        """
        logger.info(f"Processing query through security guard: {query[:50]}...")
        
        # Input validation
        validation_result = self.input_validator.validate_query(query)
        
        # Prompt injection detection
        injection_result = self.injection_detector.analyze_query(
            validation_result["sanitized_query"]
        )
        
        # Combine results
        security_analysis = {
            "is_safe": validation_result["is_valid"] and injection_result.is_safe,
            "risk_level": injection_result.risk_level,
            "validation_result": validation_result,
            "injection_result": injection_result,
            "final_query": injection_result.sanitized_query if injection_result.is_safe else "",
            "security_warnings": validation_result["warnings"] + injection_result.blocked_patterns,
            "security_errors": validation_result["errors"],
            "should_block": not (validation_result["is_valid"] and injection_result.is_safe),
            "block_reason": self._get_block_reason(validation_result, injection_result)
        }
        
        # Log security event
        self._log_security_event(query, security_analysis)
        
        return security_analysis
    
    def validate_response(self, response: str) -> Dict[str, Any]:
        """
        Validate generated response
        
        Args:
            response: Generated response to validate
            
        Returns:
            Dictionary with response validation results
        """
        return self.response_guard.validate_response(response)
    
    def _get_block_reason(self, validation_result: Dict[str, Any], 
                         injection_result: SecurityResult) -> str:
        """Get reason for blocking the query"""
        if not validation_result["is_valid"]:
            return "Input validation failed: " + "; ".join(validation_result["errors"])
        elif not injection_result.is_safe:
            return f"Security risk detected: {injection_result.warning_message}"
        else:
            return "Unknown security issue"
    
    def _log_security_event(self, query: str, analysis: Dict[str, Any]) -> None:
        """Log security event for monitoring"""
        log_entry = {
            "timestamp": logging.time.time(),
            "query_length": len(query),
            "is_safe": analysis["is_safe"],
            "risk_level": analysis["risk_level"],
            "should_block": analysis["should_block"],
            "warnings_count": len(analysis["security_warnings"]),
            "errors_count": len(analysis["security_errors"])
        }
        
        self.security_log.append(log_entry)
        
        # Keep only recent logs (last 1000 events)
        if len(self.security_log) > 1000:
            self.security_log = self.security_log[-1000:]
        
        # Log high-risk events
        if analysis["risk_level"] in ["medium", "high"]:
            logger.warning(f"Security event - Risk: {analysis['risk_level']}, Blocked: {analysis['should_block']}")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        if not self.security_log:
            return {"message": "No security events logged"}
        
        total_events = len(self.security_log)
        blocked_events = sum(1 for event in self.security_log if event["should_block"])
        high_risk_events = sum(1 for event in self.security_log if event["risk_level"] == "high")
        medium_risk_events = sum(1 for event in self.security_log if event["risk_level"] == "medium")
        
        return {
            "total_events": total_events,
            "blocked_events": blocked_events,
            "block_rate": blocked_events / total_events if total_events > 0 else 0,
            "high_risk_events": high_risk_events,
            "medium_risk_events": medium_risk_events,
            "recent_events": self.security_log[-10:]
        }
    
    def reset_security_log(self) -> None:
        """Reset security log"""
        self.security_log = []
        logger.info("Security log reset")
