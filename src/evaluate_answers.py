import re

try:
    from .rag import ask_question
except ImportError:
    from rag import ask_question


# ==================================================
# 1. TEST CASES
# ==================================================

TEST_CASES = [

    {
        "id": "A1",
        "question": "How many annual paid leave days do full-time employees receive?",

        "answerable": True,

        "expected_facts": [
            ["20 days"],
            ["annual paid leave", "annual leave"]
        ],

        "expected_sources": [
            "employee_policy.txt"
        ]
    },

    {
        "id": "A2",
        "question": "How many paid sick leave days are provided annually?",

        "answerable": True,

        "expected_facts": [
            ["10 days"],
            ["paid sick leave", "sick leave"]
        ],

        "expected_sources": [
            "employee_policy.txt"
        ]
    },

    {
        "id": "A3",
        "question": "When are employee salaries credited?",

        "answerable": True,

        "expected_facts": [
            ["last working day"],
            ["every month", "monthly"]
        ],

        "expected_sources": [
            "employee_policy.txt"
        ]
    },

    {
        "id": "A4",
        "question": "Which database technologies does the candidate know?",

        "answerable": True,

        "expected_facts": [
            ["postgresql"],
            ["mysql"],
            ["sql server"]
        ],

        "expected_sources": [
            "Misab_Abdul_Rashid_Resume.pdf"
        ]
    },

    {
        "id": "A5",
        "question": "Who is the company's CEO?",

        "answerable": False,

        "expected_facts": [],

        "expected_sources": []
    },

    {
        "id": "A6",
        "question": "What is the company's annual revenue?",

        "answerable": False,

        "expected_facts": [],

        "expected_sources": []
    }

]


# ==================================================
# 2. REFUSAL INDICATORS
# ==================================================

REFUSAL_INDICATORS = [

    "could not find",

    "not available",

    "does not contain",

    "not mentioned",

    "unable to find",

    "cannot find",

    "no information",

    "not provided",

    "information is unavailable",

    "i don't have enough information",

    "i do not have enough information",

    "not specified"

]


# ==================================================
# 3. TEXT NORMALIZATION
# ==================================================

def normalize_text(text):

    """
    Normalize text for reliable keyword matching.
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def contains_phrase(text, phrase):

    """
    Check a phrase using word boundaries.

    Prevents '20' from matching '120'.
    """

    normalized_text = normalize_text(text)

    normalized_phrase = normalize_text(phrase)

    pattern = (

        r"(?<![a-z0-9])"

        + re.escape(normalized_phrase)

        + r"(?![a-z0-9])"

    )

    return re.search(
        pattern,
        normalized_text
    ) is not None


# ==================================================
# 4. FACT CORRECTNESS EVALUATION
# ==================================================

def evaluate_facts(answer, test_case):

    """
    Each expected_facts entry is a group.

    At least one phrase in each group must appear
    in the generated answer.

    Example:

    [
        ["postgresql"],
        ["mysql"],
        ["sql server"]
    ]

    All three database technologies must appear.
    """

    missing_facts = []

    matched_facts = []

    for fact_group in test_case["expected_facts"]:

        matched = any(

            contains_phrase(answer, phrase)

            for phrase in fact_group

        )

        if matched:

            matched_facts.append(
                fact_group
            )

        else:

            missing_facts.append(
                fact_group
            )

    passed = len(missing_facts) == 0

    return {

        "passed": passed,

        "matched_facts": matched_facts,

        "missing_facts": missing_facts

    }


# ==================================================
# 5. REFUSAL EVALUATION
# ==================================================

def evaluate_refusal(answer):

    """
    Check whether the model explicitly refuses
    an unsupported question.
    """

    answer_lower = normalize_text(answer)

    matched_indicators = [

        phrase

        for phrase in REFUSAL_INDICATORS

        if contains_phrase(
            answer_lower,
            phrase
        )

    ]

    return {

        "passed": len(matched_indicators) > 0,

        "matched_indicators": matched_indicators

    }


# ==================================================
# 6. SOURCE EXTRACTION
# ==================================================

def get_source_name(source):

    """
    Extract source filename from source metadata.
    """

    if isinstance(source, dict):

        return str(

            source.get("source")

            or source.get("filename")

            or source.get("file_name")

            or ""

        )

    return str(source)


def get_source_id(source, index):

    """
    Extract citation ID.

    Falls back to source position because the RAG
    pipeline labels its context as Source 1, Source 2...
    """

    if isinstance(source, dict):

        citation_id = source.get("citation_id")

        if citation_id is not None:

            match = re.search(
                r"\d+",
                str(citation_id)
            )

            if match:
                return int(match.group())

    return index + 1


# ==================================================
# 7. CITATION EXTRACTION
# ==================================================

def extract_citations(answer):

    """
    Extract citation references such as:

    [Source 1]
    [Source 2]
    """

    matches = re.findall(

        r"\[Source\s+(\d+)\]",

        answer,

        flags=re.IGNORECASE

    )

    return sorted(
        set(int(value) for value in matches)
    )


# ==================================================
# 8. CITATION EVALUATION
# ==================================================

def evaluate_citations(answer, sources, test_case):

    citations = extract_citations(answer)

    available_source_ids = {

        get_source_id(source, index)

        for index, source in enumerate(sources)

    }

    valid_citations = [

        citation

        for citation in citations

        if citation in available_source_ids

    ]

    invalid_citations = [

        citation

        for citation in citations

        if citation not in available_source_ids

    ]

    if test_case["answerable"]:

        citation_present = len(citations) > 0

        citation_valid = (

            citation_present

            and len(invalid_citations) == 0

        )

    else:

        # Citations are not required for questions
        # that the system should refuse.

        citation_present = len(citations) > 0

        citation_valid = len(invalid_citations) == 0

    return {

        "passed": (

            citation_valid

            if test_case["answerable"]

            else not citation_present

        ),

        "citations": citations,

        "valid_citations": valid_citations,

        "invalid_citations": invalid_citations,

        "available_source_ids": sorted(
            available_source_ids
        )

    }


# ==================================================
# 9. SOURCE GROUNDING EVALUATION
# ==================================================

def evaluate_grounding(sources, test_case):

    """
    Check whether retrieved sources include
    the expected document.

    This is a source-level grounding check,
    not a complete semantic entailment test.
    """

    source_names = [

        get_source_name(source).lower()

        for source in sources

    ]

    expected_sources = test_case["expected_sources"]

    if not test_case["answerable"]:

        return {

            "passed": len(sources) == 0,

            "source_names": source_names

        }

    matching_sources = [

        expected

        for expected in expected_sources

        if any(

            expected.lower() in source_name

            for source_name in source_names

        )

    ]

    return {

        "passed": len(matching_sources) > 0,

        "matching_sources": matching_sources,

        "source_names": source_names

    }


# ==================================================
# 10. EVALUATE ONE ANSWER
# ==================================================

def evaluate_answer(answer, sources, test_case):

    if not test_case["answerable"]:

        refusal_result = evaluate_refusal(answer)

        citation_result = evaluate_citations(
            answer,
            sources,
            test_case
        )

        return {

            "correctness": refusal_result["passed"],

            "grounding": len(sources) == 0,

            "citation": citation_result["passed"],

            "refusal": refusal_result["passed"],

            "details": {

                "refusal": refusal_result,

                "citation": citation_result

            }

        }

    fact_result = evaluate_facts(
        answer,
        test_case
    )

    grounding_result = evaluate_grounding(
        sources,
        test_case
    )

    citation_result = evaluate_citations(
        answer,
        sources,
        test_case
    )

    return {

        "correctness": fact_result["passed"],

        "grounding": grounding_result["passed"],

        "citation": citation_result["passed"],

        "refusal": None,

        "details": {

            "facts": fact_result,

            "grounding": grounding_result,

            "citation": citation_result

        }

    }


# ==================================================
# 11. RUN EVALUATION
# ==================================================

def run_evaluation():

    total_tests = len(TEST_CASES)

    passed_tests = 0

    failed_tests = 0

    correctness_passed = 0

    grounding_passed = 0

    citation_passed = 0

    refusal_passed = 0

    answerable_total = 0

    unanswerable_total = 0

    print("\n======================================")

    print(" MINI ENTAURA ANSWER QUALITY EVALUATION")

    print("======================================")

    for test in TEST_CASES:

        print("\n--------------------------------------")

        print("TEST:", test["id"])

        print("Question:", test["question"])

        try:

            result = ask_question(
                test["question"]
            )

        except Exception as error:

            print("\nERROR:", str(error))

            failed_tests += 1

            if test["answerable"]:
                answerable_total += 1
            else:
                unanswerable_total += 1

            continue

        answer = result.get(
            "answer",
            ""
        )

        sources = result.get(
            "sources",
            []
        )

        print("\nGenerated answer:")

        print(answer)

        evaluation = evaluate_answer(
            answer,
            sources,
            test
        )

        # ----------------------------------
        # INDIVIDUAL METRICS
        # ----------------------------------

        print("\n========== RESULTS ==========")

        print(
            "Correctness:",
            "PASS" if evaluation["correctness"] else "FAIL"
        )

        print(
            "Grounding:",
            "PASS" if evaluation["grounding"] else "FAIL"
        )

        print(
            "Citation:",
            "PASS" if evaluation["citation"] else "FAIL"
        )

        if test["answerable"]:

            answerable_total += 1

            if evaluation["correctness"]:
                correctness_passed += 1

            if evaluation["grounding"]:
                grounding_passed += 1

            if evaluation["citation"]:
                citation_passed += 1

            passed = (

                evaluation["correctness"]

                and evaluation["grounding"]

                and evaluation["citation"]

            )

        else:

            unanswerable_total += 1

            print(
                "Refusal:",
                "PASS" if evaluation["refusal"] else "FAIL"
            )

            if evaluation["refusal"]:
                refusal_passed += 1

            if evaluation["citation"]:
                citation_passed += 1

            passed = (

                evaluation["refusal"]

                and evaluation["citation"]

            )

        # ----------------------------------
        # FAILURE DETAILS
        # ----------------------------------

        details = evaluation["details"]

        if "facts" in details:

            missing = details["facts"]["missing_facts"]

            if missing:

                print(
                    "\nMissing expected facts:",
                    missing
                )

        if "citation" in details:

            citation_details = details["citation"]

            if citation_details["invalid_citations"]:

                print(
                    "Invalid citations:",
                    citation_details["invalid_citations"]
                )

            if test["answerable"] and not citation_details["citations"]:

                print("No source citations found.")

        if "grounding" in details:

            grounding_details = details["grounding"]

            if not grounding_details["passed"]:

                print(
                    "Expected source not found:",
                    test["expected_sources"]
                )

        print("\nSources:")

        for index, source in enumerate(sources):

            print(
                f"Source {get_source_id(source, index)}:",
                get_source_name(source)
            )

        if passed:

            passed_tests += 1

            print("\nOVERALL RESULT: PASS")

        else:

            failed_tests += 1

            print("\nOVERALL RESULT: FAIL")

    # ==================================================
    # 12. FINAL SUMMARY
    # ==================================================

    def percentage(value, total):

        return (

            value / total * 100

            if total else 0

        )

    print("\n======================================")

    print(" EVALUATION SUMMARY")

    print("======================================")

    print("Total tests:", total_tests)

    print("Passed:", passed_tests)

    print("Failed:", failed_tests)

    print(
        f"Overall pass rate: "
        f"{percentage(passed_tests, total_tests):.2f}%"
    )

    print("\n---------- ANSWERABLE QUESTIONS ----------")

    print(
        f"Correctness: "
        f"{correctness_passed}/{answerable_total} "
        f"({percentage(correctness_passed, answerable_total):.2f}%)"
    )

    print(
        f"Grounding: "
        f"{grounding_passed}/{answerable_total} "
        f"({percentage(grounding_passed, answerable_total):.2f}%)"
    )

    print(
        f"Citation validity: "
        f"{citation_passed}/{total_tests} "
        f"({percentage(citation_passed, total_tests):.2f}%)"
    )

    print("\n---------- UNSUPPORTED QUESTIONS ----------")

    print(
        f"Correct refusals: "
        f"{refusal_passed}/{unanswerable_total} "
        f"({percentage(refusal_passed, unanswerable_total):.2f}%)"
    )

    print("======================================")

    print(
        "\nNOTE: These are deterministic evaluation checks. "
        "They do not establish semantic correctness or "
        "prove that every generated claim is supported."
    )


# ==================================================
# 13. EXECUTION
# ==================================================

if __name__ == "__main__":

    run_evaluation()