import sqlite3

def remove_empty_rows(result):
    return [row for row in result if any(col is not None and str(col).strip() != "" for col in row)]

def grade_basic(result1, result2):
    # should have same number of columns:
    if len(result1) == 0:
        if len(result2) == 0:
            return True
        else:
            return False
    elif len(result2) == 0:
        return False
    if len(result1[0]) == len(result2[0]):
        return True

    return False

def grade_single_number(num1, num2):
    try:
        float1 = float(num1)
        float2 = float(num2)
        return abs((float1 - float2) / float1) < 1e-2
    except:
        return False

def grade_multiset(result1, result2):
    # check if length matches
    if len(result1) != len(result2):
        return False, {"message": "Number of rows do not match"}
    # sort the tuple in each row
    sorted_result1 = [tuple(sorted([str(r) for r in row])) for row in result1]
    sorted_result2 = [tuple(sorted([str(r) for r in row])) for row in result2]
    # check if the sorted results match as multisets
    if sorted(sorted_result1) != sorted(sorted_result2):
        return False, {"message": "Results do not match as multisets"}
    return True, {"message": "Results match as multisets"}

def grade_set(result1, result2):
    # sort the tuple in each row
    sorted_result1 = [tuple(sorted([str(r) for r in row])) for row in result1]
    sorted_result2 = [tuple(sorted([str(r) for r in row])) for row in result2]
    # check if the sorted results match as sets
    if set(sorted_result1) != set(sorted_result2):
        return False, {"message": "Results do not match as sets"}
    return True, {"message": "Results match as sets"}

def grade_subset(all_results, matching_results, strict_row_count: int = None, minimum_row_count: int = None):
    # remove duplicates from matching_results
    matching_results = list(set(matching_results))
    assert strict_row_count is None or minimum_row_count is None, "Only one of strict_row_count or minimum_row_count should be provided"
    if strict_row_count is not None:
        if len(matching_results) != strict_row_count:
            return False, {"message": f"Number of matching rows {len(matching_results)} does not equal required {strict_row_count}"}
    if minimum_row_count is not None:
        if len(matching_results) < minimum_row_count:
            return False, {"message": f"Number of matching rows {len(matching_results)} is less than required minimum {minimum_row_count}"}
    # check if all matching_results are in all_results
    set_all = set([tuple(sorted([str(r) for r in row])) for row in all_results])
    set_matching = set([tuple(sorted([str(r) for r in row])) for row in matching_results])
    if not set_matching.issubset(set_all):
        return False, {"message": "Not all matching rows are in the full result set"}
    return True, {"message": "All matching rows are in the full result set"}

def grade_list(result1, result2):
    # check if length matches
    if len(result1) != len(result2):
        return False, {"message": "Number of rows do not match"}
    # sort the tuple in each row
    sorted_result1 = [tuple(sorted([str(r) for r in row])) for row in result1]
    sorted_result2 = [tuple(sorted([str(r) for r in row])) for row in result2]
    if sorted_result1 != sorted_result2:
        return False, {"message": "Results do not match in order"}
    return True, {"message": "Results match in order"}

def grade(ground_truth_result, generated_result, grading_method: str = "multiset"):
    ground_truth_result = remove_empty_rows(ground_truth_result)
    generated_result = remove_empty_rows(generated_result)

    if not grade_basic(ground_truth_result, generated_result):
        return False, {"message": "Basic grading failed: number of columns does not match"}

    if len(ground_truth_result) == 1 and len(ground_truth_result[0]) == 1 and len(generated_result) == 1 and len(generated_result[0]) == 1:
        # single number comparison
        is_correct = grade_single_number(ground_truth_result[0][0], generated_result[0][0])
        if is_correct:
            return True, {"message": "Single number matches within tolerance"}

    if "multiset" in grading_method:
        return grade_multiset(ground_truth_result, generated_result)
    elif "subset" in grading_method:
        _, method, row_count = grading_method.split(",")
        if method == "=":
            return grade_subset(ground_truth_result, generated_result, strict_row_count=int(row_count))
        elif method == ">=":
            return grade_subset(ground_truth_result, generated_result, minimum_row_count=int(row_count))
        else:
            return False, {"message": f"Unknown subset method: {method}"}
    elif "list" in grading_method:
        return grade_list(ground_truth_result, generated_result)
    elif "set" in grading_method:
        return grade_set(ground_truth_result, generated_result)
    else:
        return False, {"message": f"Unknown grading method: {grading_method}"}
    