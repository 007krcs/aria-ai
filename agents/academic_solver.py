"""
ARIA — Universal Academic Solver
==================================
One agent that handles ANY academic subject with structured, step-by-step answers.

Subjects covered:
  • Mathematics    — algebra, calculus, trigonometry, statistics, geometry, proofs
  • Chemistry      — stoichiometry, balancing equations, molar mass, pH, gas laws
  • Biology        — genetics (Punnett squares), cell biology, ecology, anatomy
  • History        — cause-effect analysis, timeline reasoning, significance
  • Economics      — supply/demand, elasticity, GDP, inflation, market structures
  • Geography      — maps, climate, physical features, human geography
  • Computer Science — algorithms, Big-O, data structures, logic gates
  • Astronomy      — Kepler's laws, stellar physics, planetary motion
  • Literature     — theme analysis, literary devices, essay structure
  • General Reasoning — logic puzzles, deductive/inductive, analogy

Strategy per subject:
  • Exact sciences  → formula library + SymPy numeric solve + step-by-step
  • Social sciences → structured concept map + LLM narrative explanation
  • Reasoning       → formal logic steps + conclusion
"""

from __future__ import annotations

import re
import math
from typing import Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# SUBJECT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

SUBJECT_KEYWORDS: dict[str, set[str]] = {
    "mathematics": {
        "derivative","integral","differentiate","integrate","limit","infinity",
        "matrix","determinant","eigenvalue","vector","dot product","cross product",
        "polynomial","quadratic","cubic","equation","solve for x","algebra",
        "trigonometry","sin","cos","tan","pythagoras","hypotenuse","triangle",
        "probability","permutation","combination","factorial","binomial",
        "statistics","mean","median","mode","standard deviation","variance",
        "geometry","area","perimeter","volume","surface area","circle","sphere",
        "logarithm","log","exponential","proof","induction","modulo",
        "sequence","series","arithmetic","geometric","convergence",
    },
    "chemistry": {
        "mole","molar mass","molarity","molecule","atom","element","compound",
        "reaction","reactant","product","stoichiometry","balance","equation",
        "acid","base","pH","pOH","buffer","titration","neutralization",
        "oxidation","reduction","redox","electron","valence","bond","ionic",
        "covalent","periodic table","element","atomic number","atomic mass",
        "enthalpy","entropy","gibbs","thermochemistry","hess law","calorimetry",
        "gas law","boyle","charles","avogadro","ideal gas","partial pressure",
        "concentration","dilution","solution","solubility","precipitation",
        "organic","alkane","alkene","alkyne","functional group","ester","amine",
        "electrochemistry","cell potential","faraday","electrolysis",
    },
    "biology": {
        "cell","nucleus","mitochondria","chloroplast","membrane","organelle",
        "dna","rna","gene","chromosome","allele","genotype","phenotype",
        "dominant","recessive","punnett","mendel","inheritance","mutation",
        "evolution","natural selection","adaptation","species","population",
        "ecosystem","food chain","food web","photosynthesis","respiration",
        "mitosis","meiosis","cell division","stem cell","tissue","organ",
        "nervous system","neuron","synapse","hormone","endocrine","immune",
        "protein","amino acid","enzyme","substrate","catalyst","metabolism",
        "osmosis","diffusion","active transport","transpiration",
        "classification","kingdom","phylum","class","order","family","genus",
    },
    "history": {
        "war","revolution","empire","civilization","ancient","medieval",
        "renaissance","industrial revolution","world war","cold war",
        "independence","colonialism","slavery","democracy","monarchy",
        "constitution","treaty","battle","invasion","conquest","dynasty",
        "pharaoh","roman","greek","persian","ottoman","mughal","british",
        "american","french","chinese","indian","african","middle ages",
        "causes of","effects of","significance of","timeline","century",
        "who was","when did","why did","how did","historical context",
        "primary source","secondary source","historian","archaeology",
    },
    "economics": {
        "supply","demand","equilibrium","price","quantity","market",
        "elasticity","gdp","inflation","deflation","recession","growth",
        "fiscal policy","monetary policy","interest rate","central bank",
        "unemployment","poverty","inequality","gini","lorenz",
        "opportunity cost","marginal","utility","consumer","producer",
        "monopoly","oligopoly","competition","market structure","game theory",
        "trade","tariff","quota","balance of payments","exchange rate",
        "comparative advantage","absolute advantage","globalization",
        "keynesian","classical","marxist","microeconomics","macroeconomics",
    },
    "geography": {
        "continent","ocean","river","mountain","climate","biome","ecosystem",
        "latitude","longitude","timezone","capital","country","population",
        "urbanization","migration","culture","language","religion",
        "plate tectonics","earthquake","volcano","erosion","weathering",
        "atmosphere","hydrosphere","lithosphere","biosphere","carbon cycle",
        "monsoon","drought","flood","hurricane","tornado","weather","map",
        "topography","physical geography","human geography",
    },
    "computer_science": {
        "algorithm","big o","time complexity","space complexity","data structure",
        "array","linked list","stack","queue","tree","binary tree","heap","graph",
        "sorting","searching","binary search","bubble sort","merge sort","quicksort",
        "recursion","dynamic programming","greedy","divide and conquer",
        "hash table","hash map","collision","load factor",
        "operating system","process","thread","deadlock","semaphore","mutex",
        "network","tcp","udp","http","https","dns","ip","routing","protocol",
        "database","sql","nosql","normalization","index","query","transaction",
        "object oriented","class","inheritance","polymorphism","encapsulation",
        "compiler","interpreter","parser","lexer","grammar","automata",
        "machine learning","neural network","gradient descent","backpropagation",
        "bit","byte","binary","hexadecimal","logic gate","and","or","nand","xor",
    },
    "astronomy": {
        "planet","star","galaxy","universe","solar system","orbit","kepler",
        "gravitational","black hole","neutron star","supernova","nebula",
        "light year","parsec","astronomical unit","redshift","blueshift",
        "telescope","spectroscopy","wavelength","electromagnetic spectrum",
        "moon","sun","mercury","venus","earth","mars","jupiter","saturn",
        "uranus","neptune","asteroid","comet","meteor","constellation",
        "big bang","dark matter","dark energy","hubble","cosmic",
    },
    "literature": {
        "poem","novel","story","character","plot","theme","setting","conflict",
        "protagonist","antagonist","metaphor","simile","personification",
        "alliteration","onomatopoeia","irony","satire","symbolism","allegory",
        "shakespeare","literary device","narrative","tone","mood","point of view",
        "essay","thesis","argument","paragraph","introduction","conclusion",
        "analyze","interpret","meaning","significance","author","text",
        "genre","fiction","nonfiction","poetry","drama","tragedy","comedy",
    },
    "reasoning": {
        "logic","logical","deduce","infer","conclude","premise","argument",
        "valid","invalid","sound","fallacy","contradiction","paradox",
        "if then","therefore","because","implies","all","some","none",
        "puzzle","riddle","brain teaser","lateral thinking","pattern",
        "analogy","comparison","relationship","category","classify",
        "syllogism","modus ponens","modus tollens","contrapositive",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MATH FORMULA LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

MATH_FORMULAS = {
    "quadratic": {
        "name": "Quadratic Formula",
        "latex": "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
        "description": "Solves ax² + bx + c = 0",
    },
    "pythagorean": {
        "name": "Pythagorean Theorem",
        "latex": "a^2 + b^2 = c^2",
        "description": "Relates sides of a right triangle",
    },
    "circle_area": {
        "name": "Area of Circle",
        "latex": "A = \\pi r^2",
        "description": "Area given radius",
    },
    "compound_interest": {
        "name": "Compound Interest",
        "latex": "A = P\\left(1 + \\frac{r}{n}\\right)^{nt}",
        "description": "Final amount with compound interest",
    },
    "derivative_power": {
        "name": "Power Rule (Differentiation)",
        "latex": "\\frac{d}{dx}[x^n] = nx^{n-1}",
        "description": "Derivative of x^n",
    },
    "integral_power": {
        "name": "Power Rule (Integration)",
        "latex": "\\int x^n\\,dx = \\frac{x^{n+1}}{n+1} + C",
        "description": "Integral of x^n (n ≠ -1)",
    },
    "binomial": {
        "name": "Binomial Theorem",
        "latex": "(a+b)^n = \\sum_{k=0}^{n}\\binom{n}{k}a^{n-k}b^k",
        "description": "Expansion of (a+b)^n",
    },
    "bayes": {
        "name": "Bayes' Theorem",
        "latex": "P(A|B) = \\frac{P(B|A)\\cdot P(A)}{P(B)}",
        "description": "Conditional probability",
    },
}

# Chemistry constants and formulas
CHEMISTRY_CONSTANTS = {
    "avogadro": 6.022e23,
    "R_gas": 8.314,       # J/(mol·K)
    "faraday": 96485,     # C/mol
    "kw": 1e-14,          # water ion product at 25°C
}

CHEM_FORMULAS = {
    "molarity": "M = n / V",          # mol/L
    "moles_mass": "n = m / M_r",      # mass / molar mass
    "ideal_gas": "PV = nRT",
    "pH": "pH = -log10([H+])",
    "pOH": "pOH = -log10([OH-])",
    "ph_poh": "pH + pOH = 14",
    "dilution": "C1V1 = C2V2",
    "percent_yield": "yield% = (actual/theoretical) × 100",
}

# Biology genetics
GENETICS = {
    "mendelian_ratios": {
        "monohybrid_f2": "3:1 (dominant:recessive)",
        "dihybrid_f2": "9:3:3:1",
        "testcross_heterozygous": "1:1",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSAL ACADEMIC SOLVER
# ─────────────────────────────────────────────────────────────────────────────

class AcademicSolverAgent:
    """
    Detects subject, selects solving strategy, returns structured step-by-step solution.
    """

    def classify_subject(self, text: str) -> list[str]:
        """Return all matching subjects, ordered by keyword match score."""
        text_lower = text.lower()
        scores: dict[str, int] = {}
        for subject, keywords in SUBJECT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[subject] = score
        return sorted(scores, key=lambda s: -scores[s])

    def solve(self, problem: str, engine=None, peer_ctx: str = "") -> dict:
        """
        Main entry point. Returns:
        {
            "solved":     bool,
            "subject":    str,
            "markdown":   str,
            "confidence": float,
        }
        """
        subjects = self.classify_subject(problem)
        if not subjects:
            # No subject detected — still try LLM with a structured prompt
            return self._llm_solve(problem, "general", engine, peer_ctx)

        primary = subjects[0]

        # Route to subject-specific handler
        handlers = {
            "mathematics":      self._solve_math,
            "chemistry":        self._solve_chemistry,
            "biology":          self._solve_biology,
            "history":          self._solve_humanities,
            "economics":        self._solve_humanities,
            "geography":        self._solve_humanities,
            "literature":       self._solve_humanities,
            "reasoning":        self._solve_reasoning,
            "computer_science": self._solve_cs,
            "astronomy":        self._solve_humanities,
        }

        handler = handlers.get(primary, self._solve_humanities)
        result = handler(problem, primary)

        # If exact solve failed, use LLM with structured context
        if not result.get("solved") and engine:
            return self._llm_solve(problem, primary, engine, peer_ctx)

        return result

    # ── Mathematics ───────────────────────────────────────────────────────────

    def _solve_math(self, problem: str, subject: str) -> dict:
        """Attempt SymPy-powered math solve."""
        try:
            import sympy as sp
        except ImportError:
            return {"solved": False}

        p = problem.lower()

        # Quadratic detection: ax² + bx + c = 0
        quad_m = re.search(
            r"([\-\d\.]+)\s*x\s*[\^²]\s*2?\s*([+\-]\s*[\d\.]+)\s*x\s*([+\-]\s*[\d\.]+)\s*=\s*0",
            problem
        )
        if quad_m or any(k in p for k in ["quadratic","x^2","x²","ax2"]):
            return self._solve_quadratic(problem)

        # Derivative
        if any(k in p for k in ["derivative","differentiate","d/dx","dy/dx","f'(x)"]):
            return self._solve_derivative(problem)

        # Integral
        if any(k in p for k in ["integral","integrate","antiderivative","∫"]):
            return self._solve_integral(problem)

        # Statistics
        if any(k in p for k in ["mean","average","median","mode","standard deviation"]):
            return self._solve_statistics(problem)

        # Pythagorean
        if any(k in p for k in ["hypotenuse","pythagorean","right triangle","right angle"]):
            return self._solve_pythagorean(problem)

        # Permutation/Combination
        if any(k in p for k in ["permutation","combination","npr","ncr","choose","arrange"]):
            return self._solve_combinatorics(problem)

        return {"solved": False}

    def _solve_quadratic(self, problem: str) -> dict:
        try:
            import sympy as sp
            x = sp.Symbol('x')
            # Extract coefficients
            nums = re.findall(r"[\-\+]?\s*[\d\.]+", problem)
            floats = [float(n.replace(" ", "")) for n in nums if n.strip()]
            if len(floats) >= 3:
                a, b, c = floats[0], floats[1], floats[2]
                discriminant = b**2 - 4*a*c
                eq = sp.Eq(a*x**2 + b*x + c, 0)
                solutions = sp.solve(eq, x)
                disc_str = f"Δ = b² - 4ac = {b}² - 4({a})({c}) = **{discriminant:.4g}**"
                if discriminant > 0:
                    nature = "Two distinct real roots"
                elif discriminant == 0:
                    nature = "One repeated real root (equal roots)"
                else:
                    nature = "Two complex (imaginary) roots"
                steps = [
                    f"**Given equation:** {a}x² + {b}x + {c} = 0",
                    f"**Formula:** x = (-b ± √(b² - 4ac)) / 2a",
                    f"**Discriminant:** {disc_str}\n→ {nature}",
                    f"**Substituting:** x = (-{b} ± √{discriminant:.4g}) / (2×{a})",
                    f"**Solutions:** x = " + ", ".join(f"**{s}**" for s in solutions),
                ]
                md = self._format_subject_md("Mathematics — Quadratic Equation", problem, steps)
                return {"solved": True, "subject": "mathematics", "markdown": md, "confidence": 0.97}
        except Exception:
            pass
        return {"solved": False}

    def _solve_derivative(self, problem: str) -> dict:
        try:
            import sympy as sp
            x = sp.Symbol('x')
            # Extract expression after "of" or "derivative of"
            m = re.search(r"(?:of|differentiate|find)\s+([x\d\s\+\-\*\/\^\(\)\.]+)", problem, re.I)
            if m:
                expr_str = m.group(1).strip().replace("^", "**")
                expr = sp.sympify(expr_str)
                deriv = sp.diff(expr, x)
                steps = [
                    f"**Expression:** f(x) = {sp.latex(expr)}",
                    f"**Rule:** Power rule — d/dx[xⁿ] = nxⁿ⁻¹",
                    f"**Derivative:** f'(x) = {sp.latex(deriv)}",
                    f"**Simplified:** f'(x) = **{deriv}**",
                ]
                md = self._format_subject_md("Mathematics — Differentiation", problem, steps)
                return {"solved": True, "subject": "mathematics", "markdown": md, "confidence": 0.97}
        except Exception:
            pass
        return {"solved": False}

    def _solve_integral(self, problem: str) -> dict:
        try:
            import sympy as sp
            x = sp.Symbol('x')
            m = re.search(r"(?:of|integrate|find)\s+([x\d\s\+\-\*\/\^\(\)\.]+)", problem, re.I)
            if m:
                expr_str = m.group(1).strip().replace("^", "**")
                expr = sp.sympify(expr_str)
                integ = sp.integrate(expr, x)
                steps = [
                    f"**Expression:** ∫ {sp.latex(expr)} dx",
                    f"**Rule:** Power rule — ∫xⁿ dx = xⁿ⁺¹/(n+1) + C",
                    f"**Result:** {sp.latex(integ)} + C",
                ]
                md = self._format_subject_md("Mathematics — Integration", problem, steps)
                return {"solved": True, "subject": "mathematics", "markdown": md, "confidence": 0.97}
        except Exception:
            pass
        return {"solved": False}

    def _solve_statistics(self, problem: str) -> dict:
        # Extract numbers from problem
        nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", problem)]
        if not nums:
            return {"solved": False}
        n = len(nums)
        mean = sum(nums) / n
        sorted_nums = sorted(nums)
        median = (sorted_nums[n//2] if n % 2 else (sorted_nums[n//2-1] + sorted_nums[n//2]) / 2)
        from collections import Counter
        freq = Counter(nums)
        mode_vals = [k for k, v in freq.items() if v == max(freq.values())]
        variance = sum((x - mean)**2 for x in nums) / n
        std_dev = math.sqrt(variance)
        steps = [
            f"**Data set:** {nums}",
            f"**n =** {n} values",
            f"**Mean:** Σx / n = {sum(nums):.4g} / {n} = **{mean:.4g}**",
            f"**Median:** Middle value of sorted data = **{median:.4g}**",
            f"**Mode:** Most frequent value(s) = **{mode_vals}**",
            f"**Variance:** Σ(x-μ)² / n = **{variance:.4g}**",
            f"**Standard Deviation:** √variance = **{std_dev:.4g}**",
        ]
        md = self._format_subject_md("Mathematics — Statistics", problem, steps)
        return {"solved": True, "subject": "mathematics", "markdown": md, "confidence": 0.95}

    def _solve_pythagorean(self, problem: str) -> dict:
        nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", problem)]
        if len(nums) >= 2:
            p = problem.lower()
            if "hypotenuse" in p or "find c" in p:
                a, b = nums[0], nums[1]
                c = math.sqrt(a**2 + b**2)
                steps = [
                    f"**Given:** a = {a}, b = {b}",
                    f"**Formula:** a² + b² = c²",
                    f"**Substituting:** {a}² + {b}² = c²",
                    f"**Calculate:** {a**2} + {b**2} = {a**2 + b**2}",
                    f"**Answer:** c = √{a**2 + b**2:.4g} = **{c:.4g}**",
                ]
            else:
                # Find a leg
                c, b = max(nums[:2]), min(nums[:2])
                a = math.sqrt(c**2 - b**2)
                steps = [
                    f"**Given:** c = {c}, b = {b}",
                    f"**Formula:** a² = c² - b²",
                    f"**Substituting:** a² = {c**2} - {b**2} = {c**2 - b**2}",
                    f"**Answer:** a = **{a:.4g}**",
                ]
            md = self._format_subject_md("Mathematics — Pythagorean Theorem", problem, steps)
            return {"solved": True, "subject": "mathematics", "markdown": md, "confidence": 0.97}
        return {"solved": False}

    def _solve_combinatorics(self, problem: str) -> dict:
        nums = [int(n) for n in re.findall(r"\b\d+\b", problem)]
        p = problem.lower()
        if len(nums) >= 2:
            n, r = nums[0], nums[1]
            try:
                if "permutation" in p or "arrange" in p or "npr" in p:
                    result = math.factorial(n) // math.factorial(n - r)
                    formula = f"P(n,r) = n! / (n-r)! = {n}! / {n-r}! = **{result}**"
                    md = self._format_subject_md("Mathematics — Permutations", problem,
                        [f"**Given:** n={n}, r={r}", f"**Formula:** {formula}"])
                else:
                    result = math.comb(n, r)
                    formula = f"C(n,r) = n! / (r!(n-r)!) = {n}! / ({r}!×{n-r}!) = **{result}**"
                    md = self._format_subject_md("Mathematics — Combinations", problem,
                        [f"**Given:** n={n}, r={r}", f"**Formula:** {formula}"])
                return {"solved": True, "subject": "mathematics", "markdown": md, "confidence": 0.97}
            except Exception:
                pass
        return {"solved": False}

    # ── Chemistry ─────────────────────────────────────────────────────────────

    def _solve_chemistry(self, problem: str, subject: str) -> dict:
        p = problem.lower()

        if "ph" in p or "hydrogen ion" in p or "[h+]" in p:
            return self._solve_ph(problem)
        if "mole" in p or "molar mass" in p or "stoichiometry" in p:
            return self._solve_moles(problem)
        if "dilution" in p or "c1v1" in p:
            return self._solve_dilution(problem)
        if "ideal gas" in p or "pv=nrt" in p or "boyle" in p:
            return self._solve_gas(problem)
        if "percent yield" in p or "theoretical yield" in p:
            return self._solve_yield(problem)

        return {"solved": False}

    def _solve_ph(self, problem: str) -> dict:
        nums = re.findall(r"[\d\.]+(?:[eE][+\-]?\d+)?", problem)
        p = problem.lower()
        steps = ["**Formula:** pH = -log₁₀[H⁺]", "**Also:** pH + pOH = 14"]
        if nums:
            val = float(nums[0])
            if "[h+" in p or "hydrogen" in p or "concentration" in p:
                ph = -math.log10(val)
                poh = 14 - ph
                steps += [
                    f"**Given:** [H⁺] = {val} mol/L",
                    f"**pH:** -log₁₀({val}) = **{ph:.4g}**",
                    f"**pOH:** 14 - {ph:.4g} = **{poh:.4g}**",
                    f"**Solution is:** {'acidic (pH < 7)' if ph < 7 else 'basic (pH > 7)' if ph > 7 else 'neutral (pH = 7)'}",
                ]
            else:
                # Given pH, find [H+]
                h_conc = 10**(-val)
                steps += [
                    f"**Given:** pH = {val}",
                    f"**[H⁺]:** 10^(-{val}) = **{h_conc:.4g} mol/L**",
                    f"**pOH:** 14 - {val} = **{14-val:.4g}**",
                ]
        md = self._format_subject_md("Chemistry — pH Calculation", problem, steps)
        return {"solved": True, "subject": "chemistry", "markdown": md, "confidence": 0.95}

    def _solve_moles(self, problem: str) -> dict:
        nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", problem)]
        steps = [
            "**Key Relationships:**",
            "- Moles (n) = Mass (g) / Molar Mass (g/mol)",
            "- Moles (n) = Number of particles / Avogadro's number (6.022×10²³)",
            "- Moles (n) = Volume (L) × Molarity (mol/L)",
        ]
        if len(nums) >= 2:
            mass, molar_mass = nums[0], nums[1]
            moles = mass / molar_mass
            particles = moles * 6.022e23
            steps += [
                f"**Given:** Mass = {mass} g, Molar Mass = {molar_mass} g/mol",
                f"**Moles:** n = {mass}/{molar_mass} = **{moles:.4g} mol**",
                f"**Particles:** {moles:.4g} × 6.022×10²³ = **{particles:.4g}**",
            ]
        md = self._format_subject_md("Chemistry — Moles & Stoichiometry", problem, steps)
        return {"solved": True, "subject": "chemistry", "markdown": md, "confidence": 0.88}

    def _solve_dilution(self, problem: str) -> dict:
        nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", problem)]
        steps = ["**Formula:** C₁V₁ = C₂V₂"]
        if len(nums) >= 3:
            known = nums[:3]
            unknown_idx = None
            p = problem.lower()
            # Find which one to solve for
            if "c2" in p or "final concentration" in p:
                c1, v1, v2 = known[0], known[1], known[2]
                c2 = (c1 * v1) / v2
                steps += [
                    f"**Given:** C₁={c1}, V₁={v1}, V₂={v2}",
                    f"**C₂ = C₁V₁/V₂ = ({c1}×{v1})/{v2} = **{c2:.4g}**",
                ]
            elif "v2" in p or "final volume" in p:
                c1, v1, c2 = known[0], known[1], known[2]
                v2 = (c1 * v1) / c2
                steps += [
                    f"**Given:** C₁={c1}, V₁={v1}, C₂={c2}",
                    f"**V₂ = C₁V₁/C₂ = ({c1}×{v1})/{c2} = **{v2:.4g} L**",
                ]
        md = self._format_subject_md("Chemistry — Dilution", problem, steps)
        return {"solved": True, "subject": "chemistry", "markdown": md, "confidence": 0.93}

    def _solve_gas(self, problem: str) -> dict:
        nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", problem)]
        R = 8.314
        steps = [
            "**Ideal Gas Law:** PV = nRT",
            f"**R =** 8.314 J/(mol·K)",
            "**Note:** Temperature must be in Kelvin (K = °C + 273.15)",
        ]
        if len(nums) >= 3:
            p_val, v_val, t_val = nums[0], nums[1], nums[2]
            # Convert Celsius to Kelvin if small temperature
            if t_val < 200:
                t_k = t_val + 273.15
                steps.append(f"**T(K):** {t_val}°C + 273.15 = {t_k} K")
            else:
                t_k = t_val
            n = (p_val * v_val) / (R * t_k)
            steps += [
                f"**Given:** P={p_val} Pa, V={v_val} m³, T={t_k} K",
                f"**n = PV/RT = ({p_val}×{v_val})/({R}×{t_k}) = **{n:.4g} mol**",
            ]
        md = self._format_subject_md("Chemistry — Ideal Gas Law", problem, steps)
        return {"solved": True, "subject": "chemistry", "markdown": md, "confidence": 0.90}

    def _solve_yield(self, problem: str) -> dict:
        nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", problem)]
        steps = ["**Formula:** % Yield = (Actual Yield / Theoretical Yield) × 100"]
        if len(nums) >= 2:
            actual, theoretical = nums[0], nums[1]
            pct = (actual / theoretical) * 100
            steps += [
                f"**Actual yield:** {actual} g",
                f"**Theoretical yield:** {theoretical} g",
                f"**% Yield = ({actual}/{theoretical}) × 100 = **{pct:.4g}%**",
            ]
        md = self._format_subject_md("Chemistry — Percent Yield", problem, steps)
        return {"solved": True, "subject": "chemistry", "markdown": md, "confidence": 0.93}

    # ── Biology ───────────────────────────────────────────────────────────────

    def _solve_biology(self, problem: str, subject: str) -> dict:
        p = problem.lower()
        if any(k in p for k in ["punnett","dominant","recessive","genotype","phenotype"]):
            return self._solve_genetics(problem)
        return {"solved": False}

    def _solve_genetics(self, problem: str) -> dict:
        p = problem.lower()
        # Detect cross type
        cross_m = re.search(r"([A-Za-z]{1,2})\s*[×x]\s*([A-Za-z]{1,2})", problem)
        if cross_m:
            p1, p2 = cross_m.group(1), cross_m.group(2)
        else:
            # Look for Aa × Aa or similar
            cross_m2 = re.search(r"([A-Za-z][a-z]?[A-Za-z][a-z]?)\s*[×x]\s*([A-Za-z][a-z]?[A-Za-z][a-z]?)", problem)
            if cross_m2:
                p1, p2 = cross_m2.group(1), cross_m2.group(2)
            else:
                p1, p2 = "Aa", "Aa"

        # Determine if monohybrid or dihybrid
        if len(p1) <= 2 and len(p2) <= 2:
            # Monohybrid
            alleles1 = list(p1)
            alleles2 = list(p2)
            combos = [a1 + a2 for a1 in alleles1 for a2 in alleles2]
            punnett_grid = [
                f"| |**{alleles2[0]}**|**{alleles2[1]}**|",
                "|---|---|---|",
                f"|**{alleles1[0]}**|{alleles1[0]+alleles2[0]}|{alleles1[0]+alleles2[1]}|",
                f"|**{alleles1[1]}**|{alleles1[1]+alleles2[0]}|{alleles1[1]+alleles2[1]}|",
            ]
            upper_count = sum(1 for c in combos if any(ch.isupper() for ch in c))
            lower_count = sum(1 for c in combos if all(ch.islower() for ch in c))
            steps = [
                f"**Cross:** {p1} × {p2} (Monohybrid cross)",
                "**Punnett Square:**\n" + "\n".join(punnett_grid),
                f"**Offspring:** {', '.join(combos)}",
                f"**Genotypic ratio:** {self._genotype_ratio(combos)}",
                f"**Phenotypic ratio:** {upper_count} dominant : {lower_count} recessive",
            ]
            md = self._format_subject_md("Biology — Genetics (Punnett Square)", problem, steps)
            return {"solved": True, "subject": "biology", "markdown": md, "confidence": 0.93}
        return {"solved": False}

    def _genotype_ratio(self, combos: list) -> str:
        from collections import Counter
        # Normalise (sort letters so Aa = Aa, aA = Aa)
        normalised = []
        for c in combos:
            if len(c) == 2:
                normalised.append("".join(sorted(c, key=lambda x: (x.lower(), x.islower()))))
            else:
                normalised.append(c)
        counts = Counter(normalised)
        return " : ".join(f"{v} {k}" for k, v in sorted(counts.items()))

    # ── Computer Science ──────────────────────────────────────────────────────

    def _solve_cs(self, problem: str, subject: str = "computer_science") -> dict:
        p = problem.lower()

        if "big o" in p or "time complexity" in p or "complexity" in p:
            return self._explain_complexity(problem)
        if "binary search" in p:
            return self._explain_algorithm(problem, "Binary Search",
                ["Check middle element", "If equal → found", "If target < middle → search left half",
                 "If target > middle → search right half", "Repeat until found or empty"],
                "O(log n)", "O(1)", "Sorted array; dramatically faster than linear search (O(n))")
        if "bubble sort" in p:
            return self._explain_algorithm(problem, "Bubble Sort",
                ["Compare adjacent elements", "Swap if in wrong order",
                 "Repeat for all elements", "After each pass, largest bubbles to end",
                 "Repeat n-1 passes total"],
                "O(n²) worst/avg, O(n) best", "O(1)", "Simple but inefficient for large datasets")
        if "merge sort" in p:
            return self._explain_algorithm(problem, "Merge Sort",
                ["Divide array in half (recursively)", "Sort each half",
                 "Merge two sorted halves", "Base case: single element is already sorted"],
                "O(n log n)", "O(n)", "Efficient, stable sort; used in Python's sorted()")
        if any(k in p for k in ["stack","queue","linked list","tree","graph","hash"]):
            return self._explain_data_structure(problem)

        return {"solved": False}

    def _explain_complexity(self, problem: str) -> dict:
        complexities = [
            ("O(1)", "Constant — accessing array element by index"),
            ("O(log n)", "Logarithmic — binary search"),
            ("O(n)", "Linear — single loop through n elements"),
            ("O(n log n)", "Linearithmic — merge sort, heap sort"),
            ("O(n²)", "Quadratic — nested loops, bubble sort"),
            ("O(2ⁿ)", "Exponential — recursive Fibonacci, subset problems"),
            ("O(n!)", "Factorial — permutations, brute-force TSP"),
        ]
        rows = "\n".join(f"| {c} | {d} |" for c, d in complexities)
        md = (
            "## Computer Science — Big-O Complexity\n\n"
            f"> {problem}\n\n"
            "### Complexity Classes (best → worst)\n\n"
            "| Big-O | Meaning |\n|---|---|\n" + rows + "\n\n"
            "### Rules for Calculating Big-O\n"
            "1. **Drop constants** — O(2n) → O(n)\n"
            "2. **Drop lower-order terms** — O(n² + n) → O(n²)\n"
            "3. **Nested loops multiply** — O(n) inside O(n) = O(n²)\n"
            "4. **Sequential steps add** — O(n) + O(n²) → O(n²)\n"
        )
        return {"solved": True, "subject": "computer_science", "markdown": md, "confidence": 0.93}

    def _explain_algorithm(self, problem: str, name: str, steps_list: list,
                           time_c: str, space_c: str, note: str) -> dict:
        steps = [
            f"**Algorithm:** {name}",
            "**Steps:**\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps_list)),
            f"**Time Complexity:** {time_c}",
            f"**Space Complexity:** {space_c}",
            f"**Note:** {note}",
        ]
        md = self._format_subject_md(f"Computer Science — {name}", problem, steps)
        return {"solved": True, "subject": "computer_science", "markdown": md, "confidence": 0.92}

    def _explain_data_structure(self, problem: str) -> dict:
        p = problem.lower()
        ds_info = {
            "stack":  ("LIFO (Last In, First Out)", "push O(1), pop O(1), peek O(1)", "Undo/redo, call stack, bracket matching"),
            "queue":  ("FIFO (First In, First Out)", "enqueue O(1), dequeue O(1)", "BFS, task scheduling, print queue"),
            "linked list": ("Nodes with next pointers", "insert O(1), search O(n), delete O(1) with pointer", "Dynamic size, no shifting needed"),
            "tree":   ("Hierarchical — parent/child nodes", "BST search O(log n) avg", "File systems, databases, HTML DOM"),
            "graph":  ("Nodes (vertices) + Edges", "DFS/BFS O(V+E)", "Social networks, maps, dependency resolution"),
            "hash":   ("Key → index via hash function", "insert/get/delete O(1) avg", "Dictionaries, caches, deduplication"),
        }
        for ds, (desc, ops, uses) in ds_info.items():
            if ds in p:
                steps = [
                    f"**Structure:** {ds.title()}",
                    f"**Definition:** {desc}",
                    f"**Operations:** {ops}",
                    f"**Use Cases:** {uses}",
                ]
                md = self._format_subject_md(f"Computer Science — {ds.title()}", problem, steps)
                return {"solved": True, "subject": "computer_science", "markdown": md, "confidence": 0.90}
        return {"solved": False}

    # ── Reasoning ─────────────────────────────────────────────────────────────

    def _solve_reasoning(self, problem: str, subject: str = "reasoning") -> dict:
        p = problem.lower()
        if "syllogism" in p or ("all" in p and "therefore" in p):
            return self._solve_syllogism(problem)
        # General logical structure
        steps = [
            "**Approach:** Break the problem into premises and conclusion",
            "**Step 1:** Identify what is given (premises)",
            "**Step 2:** Identify what is asked (conclusion)",
            "**Step 3:** Apply logical rules (deduction/induction/analogy)",
            "**Step 4:** Check for fallacies or hidden assumptions",
        ]
        md = self._format_subject_md("Logical Reasoning", problem, steps)
        return {"solved": True, "subject": "reasoning", "markdown": md, "confidence": 0.75}

    def _solve_syllogism(self, problem: str) -> dict:
        # Extract premises
        sentences = re.split(r'[.;]', problem)
        premises = [s.strip() for s in sentences if s.strip() and not any(
            k in s.lower() for k in ["therefore","conclude","so ","thus"]
        )]
        conclusion = next((s.strip() for s in sentences if any(
            k in s.lower() for k in ["therefore","conclude","so ","thus"]
        )), "")
        steps = [
            "**Type:** Categorical Syllogism",
            "**Premises:**\n" + "\n".join(f"  {i+1}. {p}" for i, p in enumerate(premises[:2])),
            f"**Conclusion:** {conclusion}",
            "**Validity check:**\n"
            "  - Does the conclusion follow necessarily from the premises?\n"
            "  - Are there any undistributed middle terms?",
        ]
        md = self._format_subject_md("Logic — Syllogism", problem, steps)
        return {"solved": True, "subject": "reasoning", "markdown": md, "confidence": 0.80}

    # ── Humanities (History, Economics, Geography, Literature, Astronomy) ─────

    def _solve_humanities(self, problem: str, subject: str) -> dict:
        """Build a structured framework for the LLM to fill in."""
        subject_prompts = {
            "history": (
                "**Framework for historical analysis:**\n"
                "1. **Context** — What was the historical situation?\n"
                "2. **Causes** — Immediate and long-term causes\n"
                "3. **Events** — Key developments in sequence\n"
                "4. **Effects** — Short-term and long-term consequences\n"
                "5. **Significance** — Why it matters today\n"
            ),
            "economics": (
                "**Framework for economic analysis:**\n"
                "1. **Define** — Key economic terms and concepts\n"
                "2. **Theory** — Relevant economic model or law\n"
                "3. **Analysis** — Apply theory to the specific scenario\n"
                "4. **Diagram** — Relevant graph (supply/demand, AS/AD, etc.)\n"
                "5. **Evaluation** — Limitations and real-world factors\n"
            ),
            "geography": (
                "**Framework for geography analysis:**\n"
                "1. **Location** — Where? Absolute and relative\n"
                "2. **Physical features** — Landforms, climate, natural resources\n"
                "3. **Human geography** — Population, economy, culture\n"
                "4. **Interactions** — How humans and environment interact\n"
                "5. **Comparisons** — Similarities/differences with other regions\n"
            ),
            "literature": (
                "**Framework for literary analysis:**\n"
                "1. **Context** — Author, era, genre\n"
                "2. **Summary** — Plot or content overview\n"
                "3. **Themes** — Central ideas and messages\n"
                "4. **Literary devices** — Techniques and their effect\n"
                "5. **Interpretation** — What does it mean? Why does it matter?\n"
            ),
            "astronomy": (
                "**Framework for astronomy:**\n"
                "1. **Object/Event** — What is being studied\n"
                "2. **Physical properties** — Size, mass, distance, temperature\n"
                "3. **Governing laws** — Relevant physics (Kepler, Newton, Einstein)\n"
                "4. **Observation** — How we detect/measure it\n"
                "5. **Significance** — Role in the universe\n"
            ),
        }
        framework = subject_prompts.get(subject, "")
        md = (
            f"## {subject.title()} — Structured Analysis\n\n"
            f"> {problem}\n\n"
            + framework +
            "\n*(Full answer requires subject knowledge — ARIA will complete this with its LLM reasoning.)*"
        )
        # Return as setup for LLM (not fully solved, but structured)
        return {
            "solved": False,
            "subject": subject,
            "setup": md,
            "llm_needed": True,
            "llm_system": (
                f"You are ARIA, an expert {subject} tutor. Answer the question below step by step. "
                f"Use the framework provided. Be thorough, accurate, and educational. "
                f"Use markdown with clear headings and bullet points."
            ),
            "confidence": 0.75,
        }

    # ── LLM fallback ─────────────────────────────────────────────────────────

    def _llm_solve(self, problem: str, subject: str, engine, peer_ctx: str = "") -> dict:
        """Use the LLM with a strong academic system prompt."""
        subject_label = subject.replace("_", " ").title()
        system = (
            f"You are ARIA, an expert {subject_label} tutor and problem solver. "
            "Solve the problem below with complete, step-by-step working. "
            "For each step: state what you are doing, show the formula or rule, "
            "apply it, and write the intermediate result. "
            "End with a clear final answer. Use markdown with headings and bullet points. "
            "Be accurate — if you're not certain, say so."
            + (f"\n\nContext from other agents:\n{peer_ctx}" if peer_ctx else "")
        )
        answer = engine.generate(
            f"Question: {problem}\n\nProvide a complete step-by-step solution:",
            system=system,
            temperature=0.15,
            max_tokens=700,
            use_cache=False,
            timeout_s=20,
        )
        if answer and len(answer) > 60:
            md = f"## {subject_label} — Step-by-Step Solution\n\n> {problem}\n\n{answer}"
            return {
                "solved": True,
                "subject": subject,
                "markdown": md,
                "confidence": 0.82,
            }
        return {"solved": False}

    # ── Formatting helper ─────────────────────────────────────────────────────

    def _format_subject_md(self, title: str, problem: str, steps: list[str]) -> str:
        lines = [
            f"## {title}",
            "",
            f"> {problem}",
            "",
        ]
        for i, step in enumerate(steps, 1):
            lines.append(f"### Step {i}")
            lines.append(step)
            lines.append("")
        lines.append("---")
        lines.append("*Solved by ARIA Academic Solver. Verify with your textbook.*")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

_academic_solver = AcademicSolverAgent()


def academic_agent_fn(query: str, engine=None, peer_ctx: str = "") -> Optional[Any]:
    """Callable for NeuralOrchestrator._get_agent_fn()."""
    try:
        from agents.omega_orchestrator import AgentResult
    except ImportError:
        return None

    result = _academic_solver.solve(query, engine=engine, peer_ctx=peer_ctx)

    if not result.get("solved"):
        # If setup was built but LLM needed
        if result.get("llm_needed") and engine:
            setup = result.get("setup", query)
            system = result.get("llm_system", "You are ARIA, an expert tutor. Solve step by step.")
            if peer_ctx:
                system += f"\n\nPeer agent context:\n{peer_ctx}"
            answer = engine.generate(
                setup, system=system, temperature=0.15, max_tokens=700, use_cache=False, timeout_s=20
            )
            if answer and len(answer) > 60:
                subject = result.get("subject", "academic")
                markdown = f"## {subject.title()} — Solution\n\n> {query}\n\n{answer}"
                return AgentResult(
                    agent="academic_solver",
                    content=markdown,
                    confidence=0.82,
                    agent_type="text",
                )
        return None

    return AgentResult(
        agent="academic_solver",
        content=result["markdown"],
        confidence=result["confidence"],
        agent_type="data" if result["confidence"] >= 0.90 else "text",
    )
