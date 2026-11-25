import re

try:
    from .rag import ask_question
except ImportError:
    from rag import ask_question


# ==================================================
# TEST CASES
# ==================================================

TEST_CASES = [

    {
        "id": "E1",
        "category": "Paraphrasing",
        "question": "What's the yearly vacation allowance for permanent employees?",
        "answerable": True,
        "required_facts": [
            ["20 days"],
            ["annual leave", "paid leave", "vacation"]
        ],
        "forbidden_claims": [],
        "expected_source": "employee_policy.txt"
    },

    {
        "id": "E2",
        "category": "Multi-part question",
        "question": "How many annual leave and sick leave days do employees receive?",
        "answerable": True,
        "required_facts": [
            ["20 days"],
            ["10 days"],
            ["annual leave", "paid leave"],
            ["sick leave"]
        ],
        "forbidden_claims": [],
        "expected_source": "employee_policy.txt"
    },

    {
        "id": "E3",
        "category": "Paraphrasing",
        "question": "On which date of the month does payroll usually reach employees?",
        "answerable": True,
        "required_facts": [
            ["last working day"]
        ],
        "forbidden_claims": [],
        "expected_source": "employee_policy.txt"
    },

    {
        "id": "E4",
        "category": "Misleading assumption",
        "question": "Full-time employees get 30 days of annual leave, correct?",
        "answerable": True,
        "required_facts": [
            ["20 days"],
            ["annual leave", "paid leave"]
        ],
        "forbidden_claims": [
            "30 days of annual leave",
            "30 days annual leave"
        ],
        "expected_source": "employee_policy.txt"
    },

    {
        "id": "E5",
        "category": "Unsupported question",
        "question": "Who founded the company?",
        "answerable": False,
        "required_facts": [],
        "forbidden_claims": [],
        "expected_source": None
    },

    {
        "id": "E6",
        "category": "Unsupported question",
        "question": "What is the company's office address?",
        "answerable": False,
        "required_facts": [],
        "forbidden_claims": [],
        "expected_source": None
    },

    {
        "id": "E7",
        "category": "Multiple facts",
        "question": "Does the employee policy mention health insurance and sick leave?",
        "answerable": True,
        "required_facts": [
            ["health insurance"],
            ["10 days"],
            ["sick leave"]
        ],
        "forbidden_claims": [],
        "expected_source": "employee_policy.txt"
    },

    {
        "id": "E8",
        "category": "Resume retrieval",
        "question": "Which relational database platforms are listed in the candidate's resume?",
        "answerable": True,
        "required_facts": [
            ["postgresql"],
            ["mysql"],
            ["sql server"]
        ],
        "forbidden_claims": [],
        "expected_source": "Misab_Abdul_Rashid_Resume.pdf"
    },

]


# ==================================================
# TEXT HELPERS
# ==================================================

def normalize(text):

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_phrase(text, phrase):

    text = normalize(text)
    phrase = normalize(phrase)

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])"
    )

    return re.search(pattern, text) is not None


# ==================================================
# FACT EVALUATION
# ==================================================

def evaluate_facts(answer, test):

    missing = []

    for group in test["required_facts"]:

        matched = any(
            contains_phrase(answer, phrase)
            for phrase in group
        )

        if not matched:
            missing.append(group)

    return missing


# ==================================================
# MISLEADING CLAIM EVALUATION
# ==================================================

def evaluate_forbidden_claims(answer, test):

    detected = []

    for claim in test["forbidden_claims"]:

        if contains_phrase(answer, claim):
            detected.append(claim)

    return detected


# ==================================================
# REFUSAL EVALUATION
# ==================================================

def evaluate_refusal(answer):

    indicators = [
        "could not find",
        "not available",
        "does not contain",
        "not mentioned",
        "unable to find",
        "cannot find",
        "no information",
        "not provided",
        "not specified",
        "i don't have enough information",
        "i do not have enough information"
    ]

    return any(
        contains_phrase(answer, phrase)
        for phrase in indicators
    )


# ==================================================
# SOURCE EVALUATION
# ==================================================

def get_source_name(source):

    if isinstance(source, dict):

        return str(
            source.get("source")
            or source.get("filename")
            or source.get("file_name")
            or ""
        )

    return str(source)


def evaluate_source(sources, test):

    expected = test["expected_source"]

    if expected is None:
        return len(sources) == 0

    return any(
        expected.lower() in get_source_name(source).lower()
        for source in sources
    )


# ==================================================
# CITATION EVALUATION
# ==================================================

def evaluate_citations(answer, sources):

    citations = re.findall(
        r"\[Source\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE
    )

    if not citations:
        return False

    valid_ids = set()

    for index, source in enumerate(sources):

        citation_id = index + 1

        if isinstance(source, dict):

            value = source.get("citation_id")

            if value is not None:

                match = re.search(r"\d+", str(value))

                if match:
                    citation_id = int(match.group())

        valid_ids.add(citation_id)

    return all(
        int(citation) in valid_ids
        for citation in citations
    )


# ==================================================
# RUN ONE TEST
# ==================================================

def run_test(test):

    print("\n" + "=" * 55)

    print(test["id"], "-", test["category"])

    print("Question:", test["question"])

    result = ask_question(test["question"])

    answer = result.get("answer", "")

    sources = result.get("sources", [])

    print("\nGenerated answer:")
    print(answer)

    if test["answerable"]:

        missing = evaluate_facts(answer, test)

        forbidden = evaluate_forbidden_claims(answer, test)

        source_pass = evaluate_source(sources, test)

        citation_pass = evaluate_citations(answer, sources)

        correctness = (
            len(missing) == 0
            and len(forbidden) == 0
        )

        overall = (
            correctness
            and source_pass
            and citation_pass
        )

        print("\nCorrectness:", "PASS" if correctness else "FAIL")

        print("Source:", "PASS" if source_pass else "FAIL")

        print("Citation:", "PASS" if citation_pass else "FAIL")

        if missing:
            print("Missing facts:", missing)

        if forbidden:
            print("Unsupported claims detected:", forbidden)

    else:

        refusal_pass = evaluate_refusal(answer)

        source_pass = len(sources) == 0

        overall = refusal_pass and source_pass

        print("\nRefusal:", "PASS" if refusal_pass else "FAIL")

        print("No irrelevant sources:", "PASS" if source_pass else "FAIL")

    print("\nRetrieved sources:")

    for source in sources:
        print("-", get_source_name(source))

    print("\nOVERALL:", "PASS" if overall else "FAIL")

    return overall


# ==================================================
# MAIN EVALUATION
# ==================================================

def run_evaluation():

    passed = 0
    failed = 0

    print("\n======================================")
    print(" MINI ENTAURA EDGE CASE EVALUATION")
    print("======================================")

    for test in TEST_CASES:

        try:

            if run_test(test):
                passed += 1
            else:
                failed += 1

        except Exception as error:

            failed += 1

            print("\nTEST ERROR:", test["id"])
            print(error)

    total = passed + failed

    accuracy = (
        passed / total * 100
        if total else 0
    )

    print("\n======================================")
    print(" FINAL EDGE CASE RESULTS")
    print("======================================")

    print("Total tests:", total)
    print("Passed:", passed)
    print("Failed:", failed)

    print(f"Pass rate: {accuracy:.2f}%")

    print("======================================")


if __name__ == "__main__":
    run_evaluation()