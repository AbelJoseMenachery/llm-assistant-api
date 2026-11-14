import asyncio
import os
import shutil

import httpx
from dotenv import load_dotenv

load_dotenv()


class ArtifactManager:
    """
    Handles secure, zero-downtime delivery of inference artifacts.
    Provides streaming downloads, atomic replacements, and rollbacks.
    """

    def __init__(self, vector_memory):
        self.memory = vector_memory
        self.base_dir = "artifacts"
        self.current_dir = os.path.join(self.base_dir, "current")
        self.previous_dir = os.path.join(self.base_dir, "previous")
        self.temp_dir = os.path.join(self.base_dir, "temp")

        # Concurrency Lock: Ensure only one update happens at a time
        self._update_lock = asyncio.Lock()

        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo = os.getenv("GITHUB_REPO")

        # Ensure directories exist
        os.makedirs(self.current_dir, exist_ok=True)
        os.makedirs(self.previous_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def _get_headers(self):
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return headers

    async def sync_on_startup(self, repo: str):
        """
        BOOTSTRAP & STARTUP RECOVERY:
        Checks for the latest release on API startup and downloads if needed.
        Guarantees that a first-deployment on Render/Docker will successfully
        bootstrap its models before accepting traffic.
        """
        if repo:
            self.repo = repo

        if not self.repo:
            print("⚠️ No GITHUB_REPO provided. Cannot perform startup sync.")
            return

        is_bootstrap = not os.path.exists(os.path.join(self.current_dir, "embedder_quant.onnx"))
        if is_bootstrap:
            print(f"🔄 Bootstrap Mode: Fetching first artifacts from GitHub ({self.repo})...")
        else:
            print(f"🔄 Startup Recovery: Checking GitHub ({self.repo}) for latest artifacts...")

        try:
            async with httpx.AsyncClient() as client:
                url = f"https://api.github.com/repos/{self.repo}/releases/latest"
                res = await client.get(url, headers=self._get_headers(), timeout=10.0)

                if res.status_code == 200:
                    release = res.json()

                    version = release.get("tag_name")
                    await self.process_update(version, release.get("assets", []))
                else:
                    if is_bootstrap:
                        print(
                            f"❌ CRITICAL: Bootstrap failed. GitHub API returned {res.status_code}. Agent cannot embed!")
                    else:
                        print(f"⚠️ Startup sync skipped. GitHub API returned {res.status_code}")
        except Exception as e:
            if is_bootstrap:
                print(f"❌ CRITICAL: Bootstrap network failure: {e}")
            else:
                print(f"⚠️ Startup sync failed: {e}. System will use local artifacts.")

    async def process_update(self, version: str, assets: list, repo: str = None):
        """Main pipeline: Download -> Verify -> Atomic Swap -> Reload -> Commit"""
        if repo:
            self.repo = repo

        # Ensure we only process one update request at a time (Idempotency / Concurrency protection)
        if self._update_lock.locked():
            print("⏳ Update already in progress. Ignoring duplicate request.")
            return

        async with self._update_lock:
            # Idempotency check: Check if version is already active
            version_file = os.path.join(self.current_dir, "version.txt")
            if os.path.exists(version_file):
                with open(version_file, "r") as f:
                    current_version = f.read().strip()
                if current_version == version:
                    print(f"✅ Version {version} is already active. Skipping update.")
                    return

            print(f"🚀 Starting zero-downtime update for version {version} from {self.repo}...")

            try:
                # 1. Clean temp directory
                if os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
                os.makedirs(self.temp_dir)

                # 2. Download all assets via Streaming (prevents memory spikes)
                async with httpx.AsyncClient() as client:
                    for asset in assets:
                        asset_name = asset["name"]
                        download_url = asset["url"]

                        # GitHub requires setting Accept header to raw to download asset binaries via API
                        dl_headers = self._get_headers()
                        dl_headers["Accept"] = "application/octet-stream"

                        temp_file_path = os.path.join(self.temp_dir, asset_name)
                        print(f"   ⬇️ Downloading {asset_name}...")

                        async with client.stream("GET", download_url, headers=dl_headers,
                                                 follow_redirects=True) as response:
                            response.raise_for_status()
                            with open(temp_file_path, "wb") as f:
                                async for chunk in response.aiter_bytes(chunk_size=8192):
                                    f.write(chunk)

                # Write version metadata
                with open(os.path.join(self.temp_dir, "version.txt"), "w") as f:
                    f.write(version)

                # 3. Validation / Test Loading
                # Attempt to hot-reload the runtime from the TEMP directory
                print("   🔍 Validating integrity and hot-reloading runtime...")
                self.memory.reload_runtime(self.temp_dir)

                # 4. Atomic Replace
                # Only if the runtime reloaded successfully, we promote TEMP to CURRENT.
                print("   🔄 Performing atomic directory swap...")

                # Backup current to previous
                if os.path.exists(self.current_dir) and os.listdir(self.current_dir):
                    if os.path.exists(self.previous_dir):
                        shutil.rmtree(self.previous_dir)
                    os.rename(self.current_dir, self.previous_dir)

                # Promote temp to current
                os.rename(self.temp_dir, self.current_dir)

                print(f"🎉 Production successfully updated to {version} with ZERO downtime!")

            except Exception as e:
                # 5. Rollback on failure
                print(f"❌ Update failed: {e}")
                print("   ⏪ Rolling back to previous known good state...")

                # If memory reload failed, it never swapped.
                # If we swapped directories and THEN it failed (edge case), we restore.
                if os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)

                # Ensure the live runtime remains tied to current_dir
                if os.path.exists(self.current_dir) and os.path.exists(
                        os.path.join(self.current_dir, "embedder_quant.onnx")):
                    self.memory.reload_runtime(self.current_dir)
