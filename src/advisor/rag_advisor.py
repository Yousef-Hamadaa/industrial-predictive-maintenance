class AIMaintenanceAdvisor:
    """
    RAG & Knowledge Base AI Maintenance Advisor.
    Provides expert guidance, failure mode descriptions, troubleshooting steps,
    and preventative action plans based on machine state.
    """

    KNOWLEDGE_BASE = {
        "twf": {
            "title": "Tool Wear Failure (TWF)",
            "description": "Tool wear failure occurs when the cutting tool wears down past its critical threshold (typically > 200–240 minutes of operational wear), increasing cutting resistance and risk of breakage.",
            "indicators": ["Tool wear > 200 min", "Elevated Torque", "High Tool Wear Strain"],
            "action_plan": [
                "1. Immediately halt active machining cycle.",
                "2. Inspect tool insert/bit for micro-cracks or plastic deformation.",
                "3. Replace worn cutting tip with standard calibrated insert.",
                "4. Reset Tool Wear timer counter in machine controller."
            ]
        },
        "hdf": {
            "title": "Heat Dissipation Failure (HDF)",
            "description": "Heat dissipation failure occurs when the difference between Process Temperature and Air Temperature drops below 8.6 K at low rotational speeds (< 1380 RPM), leading to thermal overload.",
            "indicators": ["Process Temp - Air Temp < 8.6 K", "Rotational speed < 1380 RPM"],
            "action_plan": [
                "1. Verify coolant circulation pressure and flow rate.",
                "2. Clean heat exchanger fins and thermal exhaust filters.",
                "3. Inspect temperature thermocouple sensors for calibration drift.",
                "4. Allow machine 15-minute cooling cycle before restarting."
            ]
        },
        "pwf": {
            "title": "Power Failure (PWF)",
            "description": "Power failure is triggered when mechanical power (Torque * RPM * 2*pi/60) falls outside normal working range (< 3500 W or > 9000 W), indicating stall conditions or motor overload.",
            "indicators": ["Mechanical Power < 3,500 W or > 9,000 W", "Abnormal Torque vs RPM ratio"],
            "action_plan": [
                "1. Check electrical supply voltage and motor drive inverter.",
                "2. Inspect spindle drive belt and mechanical coupling.",
                "3. Verify workpiece feed rate to avoid spindle stalling.",
                "4. Run zero-load motor diagnostic."
            ]
        },
        "osf": {
            "title": "Overstrain Failure (OSF)",
            "description": "Overstrain failure happens when the product of Tool Wear and Torque exceeds critical structural limits (varies by product variant L/M/H).",
            "indicators": ["High Tool Wear * Torque product", "Product variant structural load exceeded"],
            "action_plan": [
                "1. Reduce feed rate and depth of cut on heavy material variants (Type L/M).",
                "2. Check tool holder clamping force.",
                "3. Replace tool before reaching maximum wear limit.",
                "4. Re-evaluate CAM tooling feed parameters."
            ]
        },
        "rnf": {
            "title": "Random Failure (RNF)",
            "description": "Random failures occur independently of process parameters (~0.1% baseline probability) due to external factors.",
            "indicators": ["Spike in sensor noise", "Uncorrelated anomaly score"],
            "action_plan": [
                "1. Perform full electrical grounding test.",
                "2. Inspect vibration damper pads and mounting bolts.",
                "3. Log anomaly timestamp for baseline monitoring."
            ]
        }
    }

    def answer_query(self, query: str) -> dict:
        query_lower = query.lower()

        for key, kb in self.KNOWLEDGE_BASE.items():
            if key in query_lower or kb['title'].lower() in query_lower:
                return kb

        # General advice if query doesn't match specific failure mode
        return {
            "title": "General Machine Maintenance Guidance",
            "description": "AI4I 2020 Predictive Maintenance Platform continuously monitors 5 key sensor variables: Air Temp, Process Temp, Rotational Speed, Torque, and Tool Wear.",
            "indicators": ["Air Temp [K]", "Process Temp [K]", "Rotational Speed [RPM]", "Torque [Nm]", "Tool Wear [min]"],
            "action_plan": [
                "1. Keep Process-Air Temp difference >= 8.6 K.",
                "2. Maintain Spindle Power between 3,500 W and 9,000 W.",
                "3. Replace tools before tool wear exceeds 200 minutes.",
                "4. Monitor continuous Anomaly Score in the live dashboard."
            ]
        }

if __name__ == "__main__":
    advisor = AIMaintenanceAdvisor()
    print(advisor.answer_query("How to handle HDF failure?"))
