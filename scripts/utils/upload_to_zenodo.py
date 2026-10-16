"""Upload the data archives made by pack_data.sh to Zenodo as three draft records.

Creates the drafts (dataset, CC BY 4.0, DOI reserved), writes the DOIs of all
three parts into the descriptions and uploads the archives. Nothing is
published: check the drafts on zenodo.org and publish them there.

Large uploads to Zenodo often fail with "502 Bad Gateway", so each archive is
uploaded in pieces of 1 GiB (<name>.tar.000, <name>.tar.001, ...), several at a
time. The pieces are read directly from the archives, no extra disk space is
needed. Each piece is checked against the md5 that Zenodo computes and retried
if the upload fails. md5sums.txt (checksums of the whole archives) is uploaded
to part 1. download_data.sh joins the pieces again; by hand:
cat <name>.tar.* > <name>.tar

Running the script again continues an interrupted upload. The draft ids are
kept in zenodo_drafts.json in the archive folder, and pieces that are already
uploaded with the right checksum are skipped.

Needs a Zenodo personal access token with the scope deposit:write
(zenodo.org > Account > Applications > Personal access tokens). The script asks
for it when it starts (or reads it from the environment variable ZENODO_TOKEN).

Usage:
    python scripts/utils/upload_to_zenodo.py <archive folder> [--parallel=N] [--only=NAMES] [--sandbox] [--dry-run]

--parallel=N uploads N pieces at the same time (default 3).
--only=NAMES checks and uploads only these files (comma-separated, e.g.
    --only=results_paper.tar,md5sums.txt after repacking one archive). Without it, all
    pieces are compared with the drafts, which reads all archives once.
--sandbox uses sandbox.zenodo.org (a test instance, needs its own token).
--dry-run only prints what would be done.
"""
import concurrent.futures
import getpass
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

GITHUB = "https://github.com/MatthiasBey/intermediate-decoding"
PAPER = "Interpretability of AI weather models via intermediate decoding"
CREATORS = [{"name": "Beylich, Matthias"}, {"name": "Tempest, Kirsten"}, {"name": "Selz, Tobias"}]
KEYWORDS = ["AI weather prediction", "interpretability", "GraphCast", "Aurora",
            "ArchesWeather", "latent space", "ERA5"]

# archives of each record (each record stays below Zenodo's 50 GB and 100 files)
PARTS = {
    1: ["results_paper.tar", "data_base.tar", "data_era5.tar", "data_decodings_other.tar"],
    2: ["data_decodings_graphcast.tar", "data_latent_states.tar"],
    3: ["data_decodings_aurora.tar"],
}
PIECE = 2**30           # bytes per uploaded piece
ATTEMPTS = 20           # attempts per piece
WAIT = 120              # seconds between attempts

print_lock = threading.Lock()


def log(text):
    with print_lock:
        print(text, flush=True)


def description(part, dois):
    others = ", ".join(f"part {p}: {dois[p]}" for p in PARTS if p != part)
    text = (f"<p>Data for the paper <em>{PAPER}</em>, part {part} of 3 ({others}). "
            f'Code and instructions: <a href="{GITHUB}">{GITHUB}</a>. '
            "Use download_data.sh from the code repository to download and extract the archives.</p>"
            "<p>The archives are split into pieces of 1 GiB. To join them by hand: "
            "<code>cat &lt;name&gt;.tar.* &gt; &lt;name&gt;.tar</code>. "
            "The md5 checksums of the joined archives are in md5sums.txt in part 1.</p>")
    if part == 1:
        text += "<p>Contains modified Copernicus Climate Change Service information 2026.</p>"
    return text


def metadata(part, dois=None):
    return {
        "upload_type": "dataset",
        "title": f"Data for '{PAPER}' (part {part} of 3)",
        "creators": CREATORS,
        "description": description(part, dois) if dois else f"Data for the paper {PAPER}, part {part} of 3.",
        "license": "cc-by-4.0",
        "keywords": KEYWORDS,
        "prereserve_doi": True,
    }


def pieces(folder, part):
    """Files to upload for a record: (name on Zenodo, local file, offset, length)."""
    out = []
    for name in PARTS[part]:
        path = os.path.join(folder, name)
        size = os.path.getsize(path)
        for i in range(math.ceil(size / PIECE)):
            out.append((f"{name}.{i:03d}", path, i * PIECE, min(PIECE, size - i * PIECE)))
    if part == 1:
        path = os.path.join(folder, "md5sums.txt")
        out.append(("md5sums.txt", path, 0, os.path.getsize(path)))
    return out


class PieceReader:
    """Reads one piece of a file and computes its md5 on the way."""

    def __init__(self, path, offset, length):
        self.f = open(path, "rb")
        self.f.seek(offset)
        self.left = length
        self.md5 = hashlib.md5()

    def read(self, n=-1):
        n = self.left if n < 0 else min(n, self.left)
        data = self.f.read(n)
        self.left -= len(data)
        self.md5.update(data)
        return data

    def close(self):
        self.f.close()


def local_md5(path, offset, length):
    reader = PieceReader(path, offset, length)
    while reader.read(2**24):
        pass
    reader.close()
    return reader.md5.hexdigest()


class Zenodo:
    def __init__(self, base, token):
        self.base, self.token = base, token

    def call(self, method, path, data=None):
        url = path if path.startswith("http") else f"{self.base}/api{path}"
        body = json.dumps(data).encode() if data is not None else None
        for attempt in range(1, 6):
            req = urllib.request.Request(url, data=body, method=method, headers={
                "Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    text = r.read()
                return json.loads(text) if text else None
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    sys.exit(f"Zenodo error {e.code} for {method} {url}:\n{e.read().decode()}")
                error = f"HTTP {e.code}"
            except Exception as e:
                error = str(e)
            log(f"{method} {url} failed ({error}), attempt {attempt} of 5")
            time.sleep(30)
        sys.exit(f"Zenodo does not respond to {method} {url}. Run the script again later.")

    def put(self, url, reader, length):
        """PUT the data of reader with curl; returns (HTTP status, response body or curl error).

        curl shows the status even when Zenodo rejects the upload before all data are sent. The
        token is given to curl through a pipe, so it does not appear in the process list."""
        config_r, config_w = os.pipe()
        os.write(config_w, f'header = "Authorization: Bearer {self.token}"\n'.encode())
        os.close(config_w)
        proc = subprocess.Popen(
            ["curl", "--silent", "--show-error", "--config", f"/dev/fd/{config_r}",
             "-H", f"Content-Length: {length}", "-H", "Transfer-Encoding:",   # no chunked upload
             "-H", "Content-Type: application/octet-stream",
             "--upload-file", "-", "--write-out", "\n%{http_code}", url],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(config_r,))
        os.close(config_r)
        try:
            while data := reader.read(2**20):
                proc.stdin.write(data)
            proc.stdin.close()
        except BrokenPipeError:       # curl stopped reading, e.g. because Zenodo answered early
            pass
        out = proc.stdout.read().decode(errors="replace")
        err = proc.stderr.read().decode(errors="replace").strip()
        proc.wait()
        body, _, status = out.rpartition("\n")
        return int(status) if status.isdigit() else 0, body.strip() or err

    def upload(self, bucket, name, path, offset, length):
        """Upload one piece; returns True if Zenodo received it with the right checksum."""
        for attempt in range(1, ATTEMPTS + 1):
            start = time.time()
            reader = PieceReader(path, offset, length)
            status, body = self.put(f"{bucket}/{name}", reader, length)
            reader.close()
            if status in (200, 201):
                if json.loads(body)["checksum"].replace("md5:", "") == reader.md5.hexdigest():
                    log(f"  {name}: uploaded ({length / 1e6 / (time.time() - start):.0f} MB/s)")
                    return True
                error = "checksum does not match"
            else:
                error = f"HTTP {status}: {' '.join(body.split())[:300]}"
            log(f"  {name}: failed ({error}), attempt {attempt} of {ATTEMPTS}")
            if attempt < ATTEMPTS:
                time.sleep(WAIT)
        return False


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    folder = os.path.abspath(args[0])
    sandbox, dry_run = "--sandbox" in sys.argv, "--dry-run" in sys.argv
    parallel = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--parallel=")), 3))
    only = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--only=")), None)

    for files in PARTS.values():
        for name in files + ["md5sums.txt"]:
            if not os.path.exists(os.path.join(folder, name)):
                sys.exit(f"{name} is missing in {folder}")
    todo = {part: pieces(folder, part) for part in PARTS}
    for part, files in todo.items():
        if len(files) > 100:
            sys.exit(f"part {part} has {len(files)} files, Zenodo allows at most 100")

    if dry_run:
        for part, files in todo.items():
            size = sum(f[3] for f in files) / 1e9
            print(f"part {part} ({size:.1f} GB in {len(files)} files): {', '.join(PARTS[part])}")
            print(f"  title: {metadata(part)['title']}")
            print(f"  description: {description(part, {p: f'<DOI {p}>' for p in PARTS})}")
        return

    token = os.environ.get("ZENODO_TOKEN") or getpass.getpass("Zenodo token (input is not shown): ").strip()
    if not token:
        sys.exit("No token given.")
    zen = Zenodo("https://sandbox.zenodo.org" if sandbox else "https://zenodo.org", token)

    # create the drafts (or reuse them from an earlier run)
    state_file = os.path.join(folder, "zenodo_drafts_sandbox.json" if sandbox else "zenodo_drafts.json")
    drafts = {}
    if os.path.exists(state_file):
        with open(state_file) as f:
            drafts = {int(k): v for k, v in json.load(f).items()}
    for part in PARTS:
        if part not in drafts:
            dep = zen.call("POST", "/deposit/depositions", {"metadata": metadata(part)})
            drafts[part] = dep["id"]
            with open(state_file, "w") as f:
                json.dump(drafts, f, indent=1)
            print(f"part {part}: created draft {dep['id']}")
    deps = {part: zen.call("GET", f"/deposit/depositions/{i}") for part, i in drafts.items()}
    dois = {part: dep["metadata"]["prereserve_doi"]["doi"] for part, dep in deps.items()}

    # final metadata with the DOIs of all parts
    for part, dep in deps.items():
        zen.call("PUT", f"/deposit/depositions/{dep['id']}", {"metadata": metadata(part, dois)})

    # find out which pieces are still missing
    jobs = []
    for part, files in todo.items():
        dep = deps[part]
        draft_files = zen.call("GET", f"/deposit/depositions/{dep['id']}/files")
        uploaded = {f["filename"]: f["checksum"].replace("md5:", "") for f in draft_files}
        for extra in sorted(set(uploaded) - {f[0] for f in files}):
            print(f"part {part}: {extra} is in the draft but not needed, delete it on zenodo.org")
        checked = [f for f in files if only is None or any(f[0] == n or f[0].startswith(n + ".") for n in only)]
        missing = [f for f in checked if f[0] not in uploaded or uploaded[f[0]] != local_md5(*f[1:])]
        # files whose archive has changed since the upload: delete the old version first
        for f in draft_files:
            if f["filename"] in {name for name, *_ in missing}:
                print(f"part {part}: {f['filename']} has changed, deleting the old version in the draft")
                zen.call("DELETE", f"/deposit/depositions/{dep['id']}/files/{f['id']}")
        print(f"part {part}: {len(checked) - len(missing)} of {len(checked)} {'checked ' if only else ''}files uploaded already")
        jobs += [(dep["links"]["bucket"], *f) for f in missing]

    # upload them, several at a time
    print(f"uploading {len(jobs)} files, {parallel} at a time")
    pool = concurrent.futures.ThreadPoolExecutor(parallel)
    futures = {pool.submit(zen.upload, *job): job[1] for job in jobs}
    failed = []
    try:
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            if not future.result():
                failed.append(futures[future])
                log(f"  {futures[future]}: gave up after {ATTEMPTS} attempts, try again with the next run")
            log(f"{i - len(failed)} of {len(jobs)} uploaded, {len(failed)} failed, "
                f"{len(jobs) - i} still to do")
    except KeyboardInterrupt:
        print("\nStopped. Run the script again to continue.")
        os._exit(1)
    pool.shutdown()

    if failed:
        sys.exit(f"\n{len(failed)} files could not be uploaded: {', '.join(sorted(failed))}\n"
                 "Run the script again later, it continues where it stopped.")
    print("\nAll archives uploaded. Check the drafts and publish them on Zenodo:")
    for part, dep in deps.items():
        print(f"  part {part}: DOI {dois[part]}  {dep['links']['html']}")


if __name__ == "__main__":
    main()
