def parse_commoncrawl(wet_file):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    with open(wet_file) as f:
        lines = f.readlines()

    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]

    all_eng = ""

    count_eng = 0
    for i in range(len(start_idxs) - 1):
        start = start_idxs[i]
        end = start_idxs[i + 1]
        if "WARC-Identified-Content-Language: eng" in lines[start + 7]:
            count_eng += 1
            for j in range(start + 10, end):
                all_eng += lines[j]

    return all_eng
