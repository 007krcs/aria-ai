"""
ARIA — Physics Solver Agent
============================
Dedicated agent for solving physics problems with step-by-step working.

Capabilities:
  • Kinematics      — motion, velocity, acceleration, displacement
  • Dynamics        — force, mass, momentum, energy, work, power
  • Optics          — mirrors, lenses, refraction, Snell's law
  • Thermodynamics  — heat, temperature change, ideal gas law
  • Waves           — frequency, wavelength, period, speed
  • Electromagnetism— Ohm's law, power, Coulomb's law, circuits
  • Quantum basics  — photon energy, de Broglie wavelength

How it works:
  1. Classify the physics domain from keywords
  2. Extract known values + units from the problem text
  3. Identify what the problem is asking to find
  4. Apply the correct formula
  5. Solve symbolically with SymPy, then substitute numbers
  6. Show every step (formula → substitution → answer + unit)

Returns a full markdown step-by-step solution at confidence 0.92–0.99.
"""

from __future__ import annotations
import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# LATEX → PLAIN TEXT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _strip_latex(text: str) -> str:
    """
    Remove/convert LaTeX math notation to readable plain text.
    Called as post-processing on any LLM-generated physics output.
    """
    s = text
    # Display math blocks $$...$$ → content only
    s = re.sub(r'\$\$(.+?)\$\$', lambda m: m.group(1).strip(), s, flags=re.DOTALL)
    # Inline math $...$ → content only
    s = re.sub(r'\$(.+?)\$', lambda m: m.group(1).strip(), s)
    # Square brackets [...]  used as display math in some LLM outputs
    s = re.sub(r'^\s*\[(.+?)\]\s*$', lambda m: m.group(1).strip(), s, flags=re.MULTILINE)
    # \frac{a}{b} → (a/b)
    for _ in range(5):
        s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1/\2)', s)
    # \sqrt{x} → √x
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'√(\1)', s)
    # Superscripts ^{n} → ⁿ (digits) or ^n
    def _sup(m):
        n = m.group(1)
        digits = '⁰¹²³⁴⁵⁶⁷⁸⁹'
        return ''.join(digits[int(c)] if c.isdigit() else f'^{c}' for c in n)
    s = re.sub(r'\^\{([^{}]+)\}', _sup, s)
    s = re.sub(r'\^(\d)', lambda m: '⁰¹²³⁴⁵⁶⁷⁸⁹'[int(m.group(1))], s)
    # Subscripts _{n} → ₙ (digits)
    def _sub(m):
        n = m.group(1)
        digits = '₀₁₂₃₄₅₆₇₈₉'
        return ''.join(digits[int(c)] if c.isdigit() else c for c in n)
    s = re.sub(r'_\{([^{}]+)\}', _sub, s)
    s = re.sub(r'_(\d)', lambda m: '₀₁₂₃₄₅₆₇₈₉'[int(m.group(1))], s)
    # Common LaTeX commands
    latex_map = {
        r'\cdot': '×', r'\times': '×', r'\div': '÷', r'\pm': '±',
        r'\infty': '∞', r'\approx': '≈', r'\leq': '≤', r'\geq': '≥',
        r'\neq': '≠', r'\lambda': 'λ', r'\Lambda': 'Λ', r'\theta': 'θ',
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\Delta': 'Δ', r'\mu': 'μ', r'\pi': 'π', r'\omega': 'ω',
        r'\sigma': 'σ', r'\rho': 'ρ', r'\epsilon': 'ε',
        r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan',
        r'\boxed': '', r'\left': '', r'\right': '',
    }
    for cmd, repl in latex_map.items():
        s = s.replace(cmd, repl)
    # Remove remaining \word commands (unknown)
    s = re.sub(r'\\([a-zA-Z]+)\s*', r'\1 ', s)
    # Clean up extra spaces
    s = re.sub(r'  +', ' ', s)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# FORMULA LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

PHYSICS_DOMAINS = {
    "kinematics": {
        "keywords": {
            "velocity", "speed", "acceleration", "displacement", "distance",
            "time", "initial velocity", "final velocity", "uniformly",
            "projectile", "free fall", "gravity", "horizontal", "vertical",
            "thrown", "dropped", "launched", "motion", "position",
        },
        "formulas": [
            {
                "name": "First equation of motion",
                "equation": "v = u + a*t",
                "latex":    "v = u + at",
                "vars":     {"v": "final velocity (m/s)", "u": "initial velocity (m/s)",
                             "a": "acceleration (m/s²)", "t": "time (s)"},
            },
            {
                "name": "Second equation of motion",
                "equation": "s = u*t + 0.5*a*t**2",
                "latex":    "s = ut + \\frac{1}{2}at^2",
                "vars":     {"s": "displacement (m)", "u": "initial velocity (m/s)",
                             "a": "acceleration (m/s²)", "t": "time (s)"},
            },
            {
                "name": "Third equation of motion",
                "equation": "v**2 = u**2 + 2*a*s",
                "latex":    "v^2 = u^2 + 2as",
                "vars":     {"v": "final velocity (m/s)", "u": "initial velocity (m/s)",
                             "a": "acceleration (m/s²)", "s": "displacement (m)"},
            },
            {
                "name": "Average velocity",
                "equation": "s = ((u + v) / 2) * t",
                "latex":    "s = \\frac{u+v}{2} \\cdot t",
                "vars":     {"s": "displacement (m)", "u": "initial velocity (m/s)",
                             "v": "final velocity (m/s)", "t": "time (s)"},
            },
        ],
    },
    "dynamics": {
        "keywords": {
            "force", "mass", "newton", "acceleration", "momentum", "impulse",
            "work", "energy", "power", "kinetic", "potential", "friction",
            "weight", "normal force", "tension", "gravity", "joule", "watt",
            "newton's second", "f=ma", "ke", "pe", "gravitational",
        },
        "formulas": [
            {
                "name": "Newton's Second Law",
                "equation": "F = m * a",
                "latex":    "F = ma",
                "vars":     {"F": "force (N)", "m": "mass (kg)", "a": "acceleration (m/s²)"},
            },
            {
                "name": "Kinetic Energy",
                "equation": "KE = 0.5 * m * v**2",
                "latex":    "KE = \\frac{1}{2}mv^2",
                "vars":     {"KE": "kinetic energy (J)", "m": "mass (kg)", "v": "velocity (m/s)"},
            },
            {
                "name": "Gravitational Potential Energy",
                "equation": "PE = m * g * h",
                "latex":    "PE = mgh",
                "vars":     {"PE": "potential energy (J)", "m": "mass (kg)",
                             "g": "gravitational acceleration (9.8 m/s²)", "h": "height (m)"},
            },
            {
                "name": "Work Done",
                "equation": "W = F * d",
                "latex":    "W = Fd",
                "vars":     {"W": "work (J)", "F": "force (N)", "d": "displacement (m)"},
            },
            {
                "name": "Power",
                "equation": "P = W / t",
                "latex":    "P = \\frac{W}{t}",
                "vars":     {"P": "power (W)", "W": "work (J)", "t": "time (s)"},
            },
            {
                "name": "Momentum",
                "equation": "p = m * v",
                "latex":    "p = mv",
                "vars":     {"p": "momentum (kg·m/s)", "m": "mass (kg)", "v": "velocity (m/s)"},
            },
            {
                "name": "Weight",
                "equation": "W = m * g",
                "latex":    "W = mg",
                "vars":     {"W": "weight (N)", "m": "mass (kg)", "g": "9.8 m/s²"},
            },
        ],
    },
    "optics": {
        "keywords": {
            "lens", "mirror", "focal", "focus", "refraction", "reflection",
            "image", "object", "concave", "convex", "snell", "refractive",
            "index", "light", "optical", "diverging", "converging", "ray",
            "virtual", "real image", "magnification",
        },
        "formulas": [
            {
                "name": "Mirror/Lens Formula",
                "equation": "1/f == 1/v + 1/u",
                "latex":    "\\frac{1}{f} = \\frac{1}{v} + \\frac{1}{u}",
                "vars":     {"f": "focal length (m)", "v": "image distance (m)",
                             "u": "object distance (m)"},
                "note":     "Sign convention: distances measured from pole/optical centre",
            },
            {
                "name": "Magnification",
                "equation": "m = -v / u",
                "latex":    "m = -\\frac{v}{u}",
                "vars":     {"m": "magnification", "v": "image distance (m)", "u": "object distance (m)"},
            },
            {
                "name": "Snell's Law",
                "equation": "n1 * sin(theta1) == n2 * sin(theta2)",
                "latex":    "n_1 \\sin\\theta_1 = n_2 \\sin\\theta_2",
                "vars":     {"n1": "refractive index medium 1", "theta1": "angle of incidence (°)",
                             "n2": "refractive index medium 2", "theta2": "angle of refraction (°)"},
            },
            {
                "name": "Refractive Index",
                "equation": "n = c / v",
                "latex":    "n = \\frac{c}{v}",
                "vars":     {"n": "refractive index", "c": "speed of light (3×10⁸ m/s)",
                             "v": "speed of light in medium (m/s)"},
            },
        ],
    },
    "thermodynamics": {
        "keywords": {
            "heat", "temperature", "specific heat", "thermal", "calorimetry",
            "boiling", "melting", "gas law", "pressure", "volume", "ideal gas",
            "celsius", "kelvin", "fahrenheit", "latent heat", "entropy",
            "conduction", "convection", "radiation", "first law",
        },
        "formulas": [
            {
                "name": "Heat Transfer (Calorimetry)",
                "equation": "Q = m * c * delta_T",
                "latex":    "Q = mc\\Delta T",
                "vars":     {"Q": "heat (J)", "m": "mass (kg)", "c": "specific heat capacity (J/kg·K)",
                             "delta_T": "temperature change (K or °C)"},
            },
            {
                "name": "Ideal Gas Law",
                "equation": "P * V == n * R * T",
                "latex":    "PV = nRT",
                "vars":     {"P": "pressure (Pa)", "V": "volume (m³)", "n": "moles",
                             "R": "8.314 J/(mol·K)", "T": "temperature (K)"},
            },
            {
                "name": "Celsius to Kelvin",
                "equation": "T_K = T_C + 273.15",
                "latex":    "T(K) = T(°C) + 273.15",
                "vars":     {"T_K": "temperature in Kelvin", "T_C": "temperature in Celsius"},
            },
        ],
    },
    "waves": {
        "keywords": {
            "wave", "frequency", "wavelength", "period", "amplitude", "speed of sound",
            "sound", "light wave", "oscillation", "vibration", "hertz", "hz",
            "doppler", "interference", "diffraction", "resonance",
        },
        "formulas": [
            {
                "name": "Wave Speed",
                "equation": "v = f * lam",
                "latex":    "v = f\\lambda",
                "vars":     {"v": "wave speed (m/s)", "f": "frequency (Hz)", "lam": "wavelength (m)"},
            },
            {
                "name": "Period-Frequency Relation",
                "equation": "T = 1 / f",
                "latex":    "T = \\frac{1}{f}",
                "vars":     {"T": "period (s)", "f": "frequency (Hz)"},
            },
            {
                "name": "Photon Energy",
                "equation": "E = h * f",
                "latex":    "E = hf",
                "vars":     {"E": "energy (J)", "h": "Planck's constant (6.626×10⁻³⁴ J·s)",
                             "f": "frequency (Hz)"},
            },
        ],
    },
    "electromagnetism": {
        "keywords": {
            "voltage", "current", "resistance", "ohm", "circuit", "power",
            "electric", "charge", "coulomb", "capacitor", "inductor",
            "series", "parallel", "potential difference", "emf",
            "magnetic", "field", "flux", "faraday", "ampere", "volt",
        },
        "formulas": [
            {
                "name": "Ohm's Law",
                "equation": "V = I * R",
                "latex":    "V = IR",
                "vars":     {"V": "voltage (V)", "I": "current (A)", "R": "resistance (Ω)"},
            },
            {
                "name": "Electrical Power",
                "equation": "P = V * I",
                "latex":    "P = VI",
                "vars":     {"P": "power (W)", "V": "voltage (V)", "I": "current (A)"},
            },
            {
                "name": "Power in terms of resistance",
                "equation": "P = I**2 * R",
                "latex":    "P = I^2R",
                "vars":     {"P": "power (W)", "I": "current (A)", "R": "resistance (Ω)"},
            },
            {
                "name": "Coulomb's Law",
                "equation": "F = k * q1 * q2 / r**2",
                "latex":    "F = k\\frac{q_1 q_2}{r^2}",
                "vars":     {"F": "force (N)", "k": "8.99×10⁹ N·m²/C²",
                             "q1": "charge 1 (C)", "q2": "charge 2 (C)", "r": "distance (m)"},
            },
            {
                "name": "Energy stored in capacitor",
                "equation": "E = 0.5 * C * V**2",
                "latex":    "E = \\frac{1}{2}CV^2",
                "vars":     {"E": "energy (J)", "C": "capacitance (F)", "V": "voltage (V)"},
            },
        ],
    },
}

# Physical constants
CONSTANTS = {
    "g": 9.8,         # m/s²
    "G": 6.674e-11,   # N·m²/kg²
    "c": 3e8,         # m/s
    "h": 6.626e-34,   # J·s
    "k": 8.99e9,      # N·m²/C²
    "R": 8.314,       # J/(mol·K)
    "e": 1.6e-19,     # C
    "me": 9.11e-31,   # kg
    "Na": 6.022e23,   # mol⁻¹
}


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS SOLVER
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsSolverAgent:
    """
    Parses and solves physics problems with full step-by-step working.
    Uses SymPy for symbolic algebra; falls back to LLM for narrative framing.
    """

    # Number extraction: "5 m/s", "9.8", "3×10⁸", "3e8"
    _NUM_RE = re.compile(
        r"(?P<name>[a-zA-Z_][a-zA-Z_\s]*?)"
        r"\s*(?:=|is|of|:)\s*"
        r"(?P<val>[\d\.]+(?:[eE][+\-]?\d+)?(?:\s*[×x]\s*10\^?[\-\d]+)?)"
        r"\s*(?P<unit>[a-zA-Z/°²³·µ%]*)?",
        re.IGNORECASE
    )

    # "find", "calculate", "what is", "determine" — marks the unknown
    _FIND_RE = re.compile(
        r"(?:find|calculate|determine|what\s+is|compute|solve\s+for|"
        r"how\s+(?:fast|far|long|much|high|many)|what\s+(?:will|would|is)\s+(?:be|the))\s+"
        r"(?:the\s+)?(?P<target>[a-zA-Z][a-zA-Z\s]*?)(?:\?|$|,|\.|;)",
        re.IGNORECASE
    )

    def classify_domain(self, text: str) -> Optional[str]:
        """Return the best-matching physics domain for the problem text."""
        text_lower = text.lower()
        best_domain = None
        best_score  = 0
        for domain, info in PHYSICS_DOMAINS.items():
            score = sum(1 for kw in info["keywords"] if kw in text_lower)
            if score > best_score:
                best_score  = score
                best_domain = domain
        return best_domain if best_score > 0 else None

    def extract_values(self, text: str) -> dict[str, float]:
        """Extract named numerical values from the problem statement."""
        values: dict[str, float] = {}

        # Direct regex extraction
        for m in self._NUM_RE.finditer(text):
            name = m.group("name").strip().lower()
            val_str = m.group("val").replace(" ", "").replace("×", "e").replace("x", "e")
            try:
                val = float(val_str)
                # Normalise common names to standard variable names
                name = self._normalise_var_name(name)
                if name:
                    values[name] = val
            except ValueError:
                continue

        # Inject known constants if referenced in text
        text_lower = text.lower()
        if "gravity" in text_lower or "gravitational acceleration" in text_lower:
            values.setdefault("g", CONSTANTS["g"])
        if "speed of light" in text_lower:
            values.setdefault("c", CONSTANTS["c"])
        if "planck" in text_lower:
            values.setdefault("h", CONSTANTS["h"])

        return values

    def _normalise_var_name(self, name: str) -> Optional[str]:
        """Map natural language variable names to standard physics symbols."""
        mapping = {
            "initial velocity": "u", "initial speed": "u",
            "final velocity": "v", "final speed": "v",
            "velocity": "v", "speed": "v",
            "acceleration": "a",
            "time": "t",
            "displacement": "s", "distance": "s",
            "mass": "m",
            "force": "F",
            "height": "h",
            "frequency": "f",
            "wavelength": "lam",
            "period": "T",
            "voltage": "V", "potential difference": "V",
            "current": "I",
            "resistance": "R",
            "power": "P",
            "energy": "E",
            "heat": "Q",
            "temperature": "T", "temperature change": "delta_T",
            "pressure": "P",
            "volume": "V",
            "charge": "q",
            "object distance": "u", "image distance": "v",
            "focal length": "f",
        }
        for phrase, sym in mapping.items():
            if phrase in name:
                return sym
        # Single-letter names are likely already symbols
        if len(name.strip()) == 1 and name.strip().isalpha():
            return name.strip()
        return None

    def identify_unknown(self, text: str, known_vars: dict) -> Optional[str]:
        """Guess what variable the problem wants solved for."""
        m = self._FIND_RE.search(text)
        if m:
            target = m.group("target").strip().lower()
            mapped = self._normalise_var_name(target)
            if mapped:
                return mapped

        # Heuristic: the most commonly sought variable per domain
        text_lower = text.lower()
        if any(k in text_lower for k in ["how far", "displacement", "distance"]):
            return "s"
        if any(k in text_lower for k in ["how fast", "final velocity", "final speed"]):
            return "v"
        if any(k in text_lower for k in ["how long", "time taken"]):
            return "t"
        if "force" in text_lower:
            return "F"
        if "power" in text_lower:
            return "P"
        if "energy" in text_lower:
            return "E"
        if "heat" in text_lower:
            return "Q"
        if "resistance" in text_lower:
            return "R"
        if "current" in text_lower:
            return "I"
        if "voltage" in text_lower:
            return "V"
        return None

    def solve(self, problem: str) -> dict:
        """
        Main solver entry point.
        Returns {
            "solved":    bool,
            "domain":    str,
            "formula":   str,
            "steps":     list[str],
            "answer":    str,
            "markdown":  str,   ← full formatted solution
            "confidence": float,
        }
        """
        domain = self.classify_domain(problem)
        if not domain:
            return {"solved": False, "reason": "no_physics_domain_detected"}

        known = self.extract_values(problem)
        unknown = self.identify_unknown(problem, known)

        # Try SymPy solving
        solution = self._sympy_solve(problem, domain, known, unknown)
        if solution["solved"]:
            return solution

        # Fallback: build a structured setup for the LLM to complete
        return self._llm_setup(problem, domain, known, unknown)

    def _sympy_solve(
        self,
        problem: str,
        domain:  str,
        known:   dict[str, float],
        unknown: Optional[str],
    ) -> dict:
        """Attempt exact symbolic+numeric solve using SymPy."""
        try:
            import sympy as sp
        except ImportError:
            return {"solved": False, "reason": "sympy_not_available"}

        domain_info = PHYSICS_DOMAINS.get(domain, {})
        formulas = domain_info.get("formulas", [])

        for formula in formulas:
            vars_in_formula = list(formula["vars"].keys())
            # Check: does this formula contain the unknown?
            if unknown and unknown not in vars_in_formula:
                continue
            # Check: do we have enough known values to solve?
            missing = [v for v in vars_in_formula if v not in known and v != unknown]
            if len(missing) > 1:
                continue  # too many unknowns

            # Build SymPy equation
            try:
                syms = {v: sp.Symbol(v, real=True) for v in vars_in_formula}
                # Evaluate equation string
                eq_str = formula["equation"]

                # Handle equality vs assignment
                if "==" in eq_str:
                    lhs, rhs = eq_str.split("==", 1)
                    equation = sp.Eq(
                        sp.sympify(lhs.strip(), locals=syms),
                        sp.sympify(rhs.strip(), locals=syms),
                    )
                else:
                    lhs, rhs = eq_str.split("=", 1)
                    equation = sp.Eq(
                        sp.sympify(lhs.strip(), locals=syms),
                        sp.sympify(rhs.strip(), locals=syms),
                    )

                target_sym = syms.get(unknown or missing[0] if missing else vars_in_formula[0])
                if not target_sym:
                    continue

                # Substitute known values
                subs = {syms[k]: v for k, v in known.items() if k in syms}
                solved = sp.solve(equation.subs(subs), target_sym)
                if not solved:
                    continue

                numeric = float(solved[0].evalf())

                # Build step-by-step output
                steps = self._build_steps(formula, known, unknown or str(target_sym), numeric, domain)
                markdown = self._format_markdown(problem, domain, formula, steps, numeric, unknown)

                return {
                    "solved":     True,
                    "domain":     domain,
                    "formula":    formula["name"],
                    "steps":      steps,
                    "answer":     f"{numeric:.4g}",
                    "markdown":   markdown,
                    "confidence": 0.96,
                }
            except Exception:
                continue

        return {"solved": False, "reason": "no_matching_formula"}

    @staticmethod
    def _latex_to_plain(latex: str) -> str:
        """Convert common LaTeX math notation to readable plain text."""
        import re as _re
        s = latex
        # Fractions: \frac{a}{b} → (a/b)
        while _re.search(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', s):
            s = _re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1/\2)', s)
        # Superscripts: a^{2} → a²  or  a^2 → a²
        s = _re.sub(r'\^\{([^{}]+)\}', lambda m: ''.join(
            '⁰¹²³⁴⁵⁶⁷⁸⁹'[int(c)] if c.isdigit() else f'^{c}' for c in m.group(1)
        ), s)
        s = _re.sub(r'\^(\d)', lambda m: '⁰¹²³⁴⁵⁶⁷⁸⁹'[int(m.group(1))], s)
        # Greek letters
        replacements = {
            r'\lambda': 'λ', r'\Lambda': 'Λ',
            r'\theta': 'θ', r'\Theta': 'Θ',
            r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ',
            r'\Delta': 'Δ', r'\delta': 'δ',
            r'\mu': 'μ', r'\pi': 'π', r'\omega': 'ω',
            r'\sigma': 'σ', r'\rho': 'ρ', r'\epsilon': 'ε',
            r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan',
            r'\cdot': '×', r'\times': '×', r'\div': '÷',
            r'\pm': '±', r'\infty': '∞', r'\approx': '≈',
            r'\leq': '≤', r'\geq': '≥', r'\neq': '≠',
            r'\sqrt': '√',
        }
        for latex_cmd, plain in replacements.items():
            s = s.replace(latex_cmd, plain)
        # Subscripts: a_{1} → a₁
        s = _re.sub(r'_\{([^{}]+)\}', lambda m: ''.join(
            '₀₁₂₃₄₅₆₇₈₉'[int(c)] if c.isdigit() else c for c in m.group(1)
        ), s)
        s = _re.sub(r'_(\d)', lambda m: '₀₁₂₃₄₅₆₇₈₉'[int(m.group(1))], s)
        # Remove remaining backslashes
        s = _re.sub(r'\\([a-zA-Z]+)', r'\1', s)
        return s.strip()

    def _build_steps(
        self,
        formula: dict,
        known:   dict[str, float],
        unknown: str,
        result:  float,
        domain:  str,
    ) -> list[str]:
        steps = []
        var_desc = formula.get("vars", {})

        # Step 1: Given
        given_lines = [f"- **{k}** = {v} {var_desc.get(k,'')}" for k, v in known.items()]
        steps.append("**Given:**\n" + "\n".join(given_lines))

        # Step 2: Find
        target_desc = var_desc.get(unknown, unknown)
        steps.append(f"**Find:** {target_desc}")

        # Step 3: Formula — use plain-text notation (no LaTeX)
        plain_formula = self._latex_to_plain(formula['latex'])
        steps.append(f"**Formula ({formula['name']}):**\n`{plain_formula}`")

        # Step 4: Rearrange & substitute
        subs_parts = [f"{k} = {v}" for k, v in known.items() if k in formula["vars"]]
        steps.append("**Substituting values:**\n" + "\n".join(f"- {s}" for s in subs_parts))

        # Step 5: Answer
        unit = var_desc.get(unknown, "")
        unit_str = re.search(r"\(([^)]+)\)", unit)
        unit_label = unit_str.group(1) if unit_str else ""
        steps.append(f"**Result:** {unknown} = **{result:.4g} {unit_label}**")

        return steps

    def _format_markdown(
        self,
        problem:  str,
        domain:   str,
        formula:  dict,
        steps:    list[str],
        result:   float,
        unknown:  Optional[str],
    ) -> str:
        lines = [
            f"## Physics Solution — {domain.title()}",
            "",
            f"> **Problem:** {problem}",
            "",
        ]
        for i, step in enumerate(steps, 1):
            lines.append(f"### Step {i}")
            lines.append(step)
            lines.append("")

        # Physical plausibility note
        if unknown == "v" and result > 3e8:
            lines.append("> ⚠️ **Note:** Result exceeds speed of light — check inputs.")
        elif unknown in ("m", "t", "s") and result < 0:
            lines.append("> ⚠️ **Note:** Negative result — verify sign convention.")

        lines.append("---")
        lines.append(f"*Solved using {formula['name']}. Verify values and unit consistency.*")
        return "\n".join(lines)

    def _llm_setup(
        self,
        problem: str,
        domain:  str,
        known:   dict,
        unknown: Optional[str],
    ) -> dict:
        """Return a structured prompt setup for the LLM to complete the solution."""
        domain_info = PHYSICS_DOMAINS.get(domain, {})
        formula_list = "\n".join(
            f"- {f['name']}: {self._latex_to_plain(f['latex'])}"
            for f in domain_info.get("formulas", [])
        )
        setup = (
            f"## Physics Problem — {domain.title()}\n\n"
            f"> {problem}\n\n"
            f"**Available formulas:**\n{formula_list}\n\n"
            f"**Extracted values:** {known}\n\n"
            f"**Solve for:** {unknown or 'unknown'}\n\n"
            "Show complete step-by-step working. "
            "Use plain text math only — NO LaTeX commands (no \\frac, $$, \\cdot etc.). "
            "Write fractions as (a/b), powers as a^2, multiplication as × or *."
        )
        return {
            "solved":     False,
            "domain":     domain,
            "setup":      setup,
            "llm_needed": True,
            "confidence": 0.70,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT INTERFACE (compatible with NeuralOrchestrator)
# ─────────────────────────────────────────────────────────────────────────────

_solver = PhysicsSolverAgent()


def physics_agent_fn(query: str, engine=None, peer_ctx: str = "") -> Optional[object]:
    """
    Callable for NeuralOrchestrator._get_agent_fn().
    Returns an AgentResult or None.
    """
    try:
        from agents.omega_orchestrator import AgentResult
    except ImportError:
        return None

    result = _solver.solve(query)

    if result.get("solved"):
        return AgentResult(
            agent="physics_solver",
            content=result["markdown"],
            confidence=result["confidence"],
            agent_type="data",   # treated as a self-contained authoritative answer
        )

    if result.get("llm_needed") and engine:
        # SymPy couldn't fully solve — give LLM a structured setup
        setup = result.get("setup", "")
        system = (
            "You are ARIA, a physics tutor. Solve the following problem step by step. "
            "Use the formula list and extracted values provided. "
            "Show every step: given → formula → substitution → calculation → answer with units. "
            "Use markdown with clear sections. "
            "IMPORTANT: Use plain text math ONLY — no LaTeX (no \\frac, $$, \\cdot, ^{}, etc.). "
            "Write fractions as (a/b), powers as a^2 or use ² ³ symbols, multiplication as × or *."
            + (f"\n\n{peer_ctx}" if peer_ctx else "")
        )
        llm_answer = engine.generate(
            setup, system=system, temperature=0.1, max_tokens=600, use_cache=False, timeout_s=20
        )
        if llm_answer and len(llm_answer) > 60:
            # Post-process: strip any residual LaTeX the LLM may have emitted
            llm_answer = _strip_latex(llm_answer)
            return AgentResult(
                agent="physics_solver",
                content=f"## Physics Solution — {result.get('domain','').title()}\n\n{llm_answer}",
                confidence=0.82,
                agent_type="text",
            )

    return None
