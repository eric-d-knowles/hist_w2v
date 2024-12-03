import argparse
import orjson

def print_jsonl_line(file_path, start_line=1, end_line=5, parse_json=False):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):  # start=1 for 1-based index
                if i < start_line:
                    continue
                if i > end_line:
                    break
                if parse_json:
                    try:
                        parsed_line = orjson.loads(line)
                        print(f"Line {i}: {parsed_line}")
                    except orjson.JSONDecodeError:
                        print(f"Line {i}: Error parsing JSON: {line.strip()}")
                else:
                    print(f"Line {i}: {line.strip()}\n")
    except Exception as e:
        print(f"Error reading the file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inspect a range of lines from a JSONL file.")
    parser.add_argument("--file_path", type=str, help="Path to the JSONL file.")
    parser.add_argument("--start", type=int, default=1, help="The starting line number (1-based). Default is 1.")
    parser.add_argument("--end", type=int, default=5, help="The ending line number (inclusive). Default is 5.")
    parser.add_argument("--parse", action="store_true", help="Parse lines as JSON using orjson. Default is False (prints raw lines).")
    args = parser.parse_args()

    print_jsonl_line(
        file_path=args.file_path,
        start_line=args.start,
        end_line=args.end,
        parse_json=args.parse
    )

if __name__ == "__main__":
    main()
