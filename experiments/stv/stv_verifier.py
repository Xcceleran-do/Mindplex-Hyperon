#!/usr/bin/env python3
"""
STV Verifier for Backward Chaining Results
Enhances backward chaining proofs with STV confidence verification and aggregation.
"""

import re
import json
from typing import List, Dict, Any, Union, Optional, Tuple
from dataclasses import dataclass


@dataclass
class STVValue:
    """Represents a Single Transferable Vote truth value"""
    strength: float  # Confidence in the truth value (0.0 to 1.0)
    confidence: float  # Strength of the evidence (0.0 to 1.0)
    
    def __str__(self):
        return f"(stv {self.strength} {self.confidence})"
    
    def to_dict(self):
        return {
            "strength": self.strength,
            "confidence": self.confidence,
            "combined_score": self.strength * self.confidence
        }


@dataclass
class ProofNode:
    """Represents a single node in a proof tree"""
    rule: str
    conclusion: str
    stv: Optional[STVValue] = None
    premises: List['ProofNode'] = None
    
    def __post_init__(self):
        if self.premises is None:
            self.premises = []


@dataclass
class EnhancedProof:
    """Enhanced proof with STV verification and confidence aggregation"""
    proof_index: int
    conclusion: str
    nodes: List[ProofNode]
    aggregated_stv: STVValue
    verification_status: str
    confidence_breakdown: Dict[str, Any]


class STVVerifier:
    """Main STV verification and enhancement system"""
    
    def __init__(self):
        self.default_stv = STVValue(1.0, 1.0)  # Default for facts without STV
        self.weak_rule_threshold = 0.3
        self.strong_rule_threshold = 0.7
    
    def parse_stv_from_atom(self, atom_str: str) -> Optional[STVValue]:
        """Extract STV values from MeTTa atom strings"""
        if not atom_str or not isinstance(atom_str, str):
            return None
            
        # Look for (stv strength confidence) pattern
        stv_match = re.search(r'\(stv\s+([\d.]+)\s+([\d.]+)\)', atom_str)
        if stv_match:
            try:
                strength = float(stv_match.group(1))
                confidence = float(stv_match.group(2))
                return STVValue(strength, confidence)
            except ValueError:
                pass
        
        return None
    
    def extract_conclusion_from_atom(self, atom_str: str) -> str:
        """Extract the conclusion part from a MeTTa atom"""
        if not atom_str:
            return ""
            
        # Remove rule wrapper if present
        if "(rule:-" in atom_str:
            # Extract content after rule:-
            match = re.search(r'\(rule:-\s*(.+?)\)', atom_str)
            if match:
                atom_str = match.group(1)
        
        # For implications, extract the conclusion (right side of ->)
        if "(->" in atom_str:
            # Find the conclusion part (after the last ->)
            parts = atom_str.split("(->")
            if len(parts) >= 2:
                conclusion_part = parts[-1].rstrip(")")
                return f"(-> {conclusion_part})"
        
        return atom_str
    
    def analyze_proof_structure(self, proof: Any) -> List[ProofNode]:
        """Analyze a proof and extract nodes with STV information"""
        nodes = []
        
        if isinstance(proof, (list, tuple)):
            # Handle MeTTa proof structure: (: ((rule premise) (fact premise)) conclusion)
            if len(proof) >= 3 and str(proof[0]) == ":":
                # Extract rule and fact premises
                premises = proof[1] if len(proof) > 1 else []
                conclusion = proof[2] if len(proof) > 2 else None
                
                # Create node for this proof step
                conclusion_str = str(conclusion) if conclusion else ""
                stv = self.parse_stv_from_atom(conclusion_str)
                
                node = ProofNode(
                    rule=str(premises) if premises else "direct_fact",
                    conclusion=conclusion_str,
                    stv=stv or self.default_stv
                )
                nodes.append(node)
                
            else:
                # Handle nested structures
                for item in proof:
                    nodes.extend(self.analyze_proof_structure(item))
        
        elif isinstance(proof, str):
            # Handle string representations
            stv = self.parse_stv_from_atom(proof)
            node = ProofNode(
                rule="string_proof",
                conclusion=proof,
                stv=stv or self.default_stv
            )
            nodes.append(node)
        
        return nodes
    
    def aggregate_stv_confidence(self, nodes: List[ProofNode]) -> STVValue:
        """Aggregate STV values across proof nodes using weighted averaging"""
        if not nodes:
            return self.default_stv
        
        total_strength = 0.0
        total_confidence = 0.0
        node_count = 0
        
        for node in nodes:
            if node.stv:
                total_strength += node.stv.strength
                total_confidence += node.stv.confidence
                node_count += 1
        
        if node_count == 0:
            return self.default_stv
        
        # Use weighted averaging
        avg_strength = total_strength / node_count
        avg_confidence = total_confidence / node_count
        
        # Apply confidence decay for longer proof chains
        decay_factor = max(0.5, 1.0 - (len(nodes) - 1) * 0.1)
        final_confidence = avg_confidence * decay_factor
        
        return STVValue(avg_strength, final_confidence)
    
    def verify_proof_validity(self, nodes: List[ProofNode]) -> str:
        """Verify the logical validity of a proof"""
        if not nodes:
            return "invalid_empty"
        
        # Check for rule-fact consistency
        has_rules = any("rule:" in node.rule for node in nodes)
        has_facts = any("fact:" in node.rule or "direct_fact" in node.rule for node in nodes)
        
        if has_rules and not has_facts:
            return "warning_missing_facts"
        elif has_facts and not has_rules:
            return "direct_fact_only"
        elif not has_rules and not has_facts:
            return "unknown_structure"
        
        return "valid"
    
    def enhance_single_proof(self, proof: Any, proof_index: int) -> EnhancedProof:
        """Enhance a single proof with STV verification"""
        # Analyze proof structure
        nodes = self.analyze_proof_structure(proof)
        
        # Aggregate STV confidence
        aggregated_stv = self.aggregate_stv_confidence(nodes)
        
        # Verify proof validity
        verification_status = self.verify_proof_validity(nodes)
        
        # Create confidence breakdown
        confidence_breakdown = {
            "node_count": len(nodes),
            "nodes_with_stv": sum(1 for node in nodes if node.stv),
            "aggregated_strength": aggregated_stv.strength,
            "aggregated_confidence": aggregated_stv.confidence,
            "combined_score": aggregated_stv.strength * aggregated_stv.confidence,
            "verification_status": verification_status
        }
        
        # Extract main conclusion
        main_conclusion = nodes[-1].conclusion if nodes else ""
        
        return EnhancedProof(
            proof_index=proof_index,
            conclusion=main_conclusion,
            nodes=nodes,
            aggregated_stv=aggregated_stv,
            verification_status=verification_status,
            confidence_breakdown=confidence_breakdown
        )
    
    def verify_proofs_stv(self, chain_answer: Any) -> List[Dict[str, Any]]:
        """
        Main verification function - enhances backward chaining results with STV confidence
        
        Args:
            chain_answer: Raw backward chaining results from MeTTa
            
        Returns:
            List of enhanced proofs with STV verification data
        """
        enhanced_proofs = []
        
        try:
            # Handle different input formats
            proofs = []
            
            if isinstance(chain_answer, (list, tuple)):
                # Flat list of proofs
                proofs = list(chain_answer)
            elif hasattr(chain_answer, '__iter__'):
                # Iterable object
                proofs = list(chain_answer)
            else:
                # Single proof or unexpected format
                proofs = [chain_answer]
            
            # Process each proof
            for idx, proof in enumerate(proofs):
                try:
                    enhanced_proof = self.enhance_single_proof(proof, idx + 1)
                    
                    # Convert to dictionary for JSON serialization
                    proof_dict = {
                        "proof_index": enhanced_proof.proof_index,
                        "conclusion": enhanced_proof.conclusion,
                        "verification_status": enhanced_proof.verification_status,
                        "aggregated_stv": enhanced_proof.aggregated_stv.to_dict(),
                        "confidence_breakdown": enhanced_proof.confidence_breakdown,
                        "nodes": [
                            {
                                "rule": node.rule,
                                "conclusion": node.conclusion,
                                "stv": node.stv.to_dict() if node.stv else None
                            }
                            for node in enhanced_proof.nodes
                        ]
                    }
                    enhanced_proofs.append(proof_dict)
                    
                except Exception as e:
                    print(f"Error processing proof {idx + 1}: {e}")
                    # Add error fallback
                    enhanced_proofs.append({
                        "proof_index": idx + 1,
                        "conclusion": str(proof),
                        "verification_status": "processing_error",
                        "aggregated_stv": STVValue(0.0, 0.0).to_dict(),
                        "confidence_breakdown": {"error": str(e)},
                        "nodes": []
                    })
            
            # Add overall summary
            if enhanced_proofs:
                total_confidence = sum(p["aggregated_stv"]["combined_score"] for p in enhanced_proofs)
                avg_confidence = total_confidence / len(enhanced_proofs)
                
                summary = {
                    "total_proofs": len(enhanced_proofs),
                    "average_confidence": avg_confidence,
                    "high_confidence_proofs": sum(1 for p in enhanced_proofs 
                                               if p["aggregated_stv"]["confidence"] > self.strong_rule_threshold),
                    "low_confidence_proofs": sum(1 for p in enhanced_proofs 
                                              if p["aggregated_stv"]["confidence"] < self.weak_rule_threshold)
                }
                
                # Add summary to each proof or return separately
                for proof in enhanced_proofs:
                    proof["summary"] = summary
            
        except Exception as e:
            print(f"STV verification failed: {e}")
            # Return minimal fallback
            enhanced_proofs = [{
                "proof_index": 1,
                "conclusion": str(chain_answer),
                "verification_status": "verification_failed",
                "aggregated_stv": STVValue(0.0, 0.0).to_dict(),
                "confidence_breakdown": {"error": str(e)},
                "nodes": [],
                "summary": {"error": str(e)}
            }]
        
        return enhanced_proofs


# Main verification function that matches the import in getChainerResult
def verify_proofs_stv(chain_answer: Any) -> List[Dict[str, Any]]:
    """
    Verify and enhance backward chaining results with STV confidence
    
    This function is imported in mining_api.py and called from getChainerResult()
    
    Args:
        chain_answer: Raw backward chaining results from MeTTa backward-chain
        
    Returns:
        Enhanced proofs with STV verification, confidence scores, and validity checks
    """
    verifier = STVVerifier()
    return verifier.verify_proofs_stv(chain_answer)


# Utility functions for testing and debugging
def test_stv_verifier():
    """Test the STV verifier with sample data"""
    # Sample proof structure similar to MeTTa output
    sample_proof = [
        [":", 
         [["rule:-", ["->", ["length", "1", "low"], ["engagement", "1", "high"]]], 
          ["fact:-", ["length", "1", "low"]]], 
         ["engagement", "1", "high", ["stv", "0.8", "0.7"]]
        ]
    ]
    
    verifier = STVVerifier()
    result = verify_proofs_stv(sample_proof)
    
    print("STV Verification Test Results:")
    print(json.dumps(result, indent=2))
    
    return result


if __name__ == "__main__":
    # Run test if executed directly
    test_stv_verifier()
