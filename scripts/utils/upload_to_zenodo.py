"""Upload the data archives made by pack_data.sh to Zenodo as three draft records.

Creates the drafts (dataset, CC BY 4.0, DOI reserved), writes the DOIs of all
three parts into the descriptions and uploads the archives. Each upload is
checked against md5sums.txt. Nothing is published: check the drafts on
zenodo.org and publish them there.

Running the script again continues an interrupted upload. The draft ids are
kept in zenodo_drafts.json in the archive folder, and files that are already
uploaded with the right checksum are skipped.

Needs a Zenodo personal access token with the scope deposit:write
(zenodo.org > Account > Applications > Personal access tokens). The script asks
for it when it starts (or reads it from the environment variable ZENODO_TOKEN).

Usage:
    python scripts/utils/upload_to_zenodo.py <archive folder> [--sandbox] [--dry-run]

--sandbox uses sandbox.zenodo.org (a test instance, needs its own token).
--dry-run only prints what would be done.
"""
import getpass
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

GITHUB = "https://github.com/MatthiasBey/intermediate-decoding"
PAPER = "Interpretability of AI weather models via intermediate decoding"
CREATORS = [{"name": "Beylich, Matthias"}, {"name": "Tempest, Kirsten"}, {"name": "Selz, Tobias"}]
KEYWORDS = ["AI weather prediction", "interpretability", "GraphCast", "Aurora",
            "ArchesWeather", "latent space", "ERA5"]

# archives of each record (each record stays below Zenodo's 50 GB limit)
PARTS = {
    1: ["results_paper.tar", "data_base.tar", "data_era5.tar", "data_decodings_other.tar"],
    2: ["data_decodings_graphcast.tar", "data_latent_states.tar"],
    3: ["data_decodings_aurora.tar"],
}


def description(part, dois):
    others = ", ".join(f"part {p}: {dois[p]}" for p in PARTS if p != part)
    text = (f"<p>Data for the paper <em>{PAPER}</em>, part {part} of 3 ({others}). "
            f'Code and instructions: <a href="{GITHUB}">{GITHUB}</a>. '
            "Use download_data.sh from the code repository to download and extract the archives.</p>")
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


class Zenodo:
    def __init__(self, base, token, dry_run):
        self.base, self.token, self.dry_run = base, token, dry_run

    def call(self, method, path, data=None):
        url = path if path.startswith("http") else f"{self.base}/api{path}"
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(url, data=body, method=method, headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                text = r.read()
        except urllib.error.HTTPError as e:
            sys.exit(f"Zenodo error {e.code} for {method} {url}:\n{e.read().decode()}")
        return json.loads(text) if text else None

    def upload(self, bucket, path, attempts=5):
        """Stream a file into the draft's bucket with curl; returns the md5 Zenodo computed.

        The token is given to curl on stdin, so it does not appear in the process list or in
        error messages. Failed uploads (e.g. 502 from Zenodo) are retried."""
        name = os.path.basename(path)
        for attempt in range(1, attempts + 1):
            out = subprocess.run(
                ["curl", "--config", "-", "--fail-with-body", "--progress-bar",
                 "--upload-file", path, f"{bucket}/{name}"],
                input=f'header = "Authorization: Bearer {self.token}"\n'.encode(), stdout=subprocess.PIPE)
            if out.returncode == 0:
                return json.loads(out.stdout)["checksum"].replace("md5:", "")
            print(f"upload of {name} failed (curl exit code {out.returncode}, attempt {attempt} of {attempts})")
            if out.stdout:
                print(out.stdout.decode(errors="replace")[:500])
            if attempt < attempts:
                time.sleep(60)
        sys.exit(f"Uploading {name} failed {attempts} times. Run the script again later.")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    folder = os.path.abspath(args[0])
    sandbox, dry_run = "--sandbox" in sys.argv, "--dry-run" in sys.argv
    token = os.environ.get("ZENODO_TOKEN")
    if not token and not dry_run:
        token = getpass.getpass("Zenodo token (input is not shown): ").strip()
    if not token and not dry_run:
        sys.exit("No token given.")
    zen = Zenodo("https://sandbox.zenodo.org" if sandbox else "https://zenodo.org", token, dry_run)

    md5 = {}
    with open(os.path.join(folder, "md5sums.txt")) as f:
        for line in f:
            checksum, name = line.split()
            md5[os.path.basename(name)] = checksum
    for files in PARTS.values():
        for name in files:
            if name not in md5 or not os.path.exists(os.path.join(folder, name)):
                sys.exit(f"{name} is missing in {folder} or in md5sums.txt")

    if dry_run:
        for part, files in PARTS.items():
            size = sum(os.path.getsize(os.path.join(folder, n)) for n in files) / 1e9
            print(f"part {part} ({size:.1f} GB): {', '.join(files)}")
            print(f"  title: {metadata(part)['title']}")
            print(f"  description: {description(part, {p: f'<DOI {p}>' for p in PARTS})}")
        return

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

    # upload the archives
    for part, files in PARTS.items():
        dep = deps[part]
        uploaded = {f["filename"]: f["checksum"] for f in zen.call("GET", f"/deposit/depositions/{dep['id']}/files")}
        for name in files:
            if uploaded.get(name) == md5[name]:
                print(f"part {part}: {name} already uploaded")
                continue
            print(f"part {part}: uploading {name}")
            checksum = zen.upload(dep["links"]["bucket"], os.path.join(folder, name))
            if checksum != md5[name]:
                sys.exit(f"Checksum of the uploaded {name} does not match md5sums.txt. Run the script again.")

    print("\nAll archives uploaded. Check the drafts and publish them on Zenodo:")
    for part, dep in deps.items():
        print(f"  part {part}: DOI {dois[part]}  {dep['links']['html']}")


if __name__ == "__main__":
    main()
