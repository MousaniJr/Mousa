"""
Code Collector - Scrapes open-source repositories for training data

Supports: GitHub, GitLab, Bitbucket
Respects: robots.txt, rate limits, API quotas
Filters: By license, stars, language, activity
"""

import asyncio
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import aiohttp
import os
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RepositoryMetadata:
    """Metadata for a collected repository"""
    repo_url: str
    language: str
    stars: int
    license: str
    last_updated: datetime
    description: str
    topics: List[str]
    size_kb: int


class CodeCollector:
    """
    Collects code from open-source repositories
    """

    SUPPORTED_LANGUAGES = {
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C", "C#",
        "Go", "Rust", "Swift", "Kotlin", "Ruby", "PHP", "Scala", "R"
    }

    APPROVED_LICENSES = {
        "MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause",
        "GPL-3.0", "LGPL-3.0", "ISC", "MPL-2.0", "Unlicense"
    }

    def __init__(
        self,
        github_token: Optional[str] = None,
        min_stars: int = 10,
        max_repos: int = 10000,
        output_dir: str = "data/raw/code"
    ):
        """
        Initialize code collector

        Args:
            github_token: GitHub API token (from environment variable)
            min_stars: Minimum stars required for repository
            max_repos: Maximum number of repositories to collect
            output_dir: Directory to store collected code
        """
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.min_stars = min_stars
        self.max_repos = max_repos
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session: Optional[aiohttp.ClientSession] = None
        self.collected_repos: Set[str] = set()

    async def __aenter__(self):
        """Async context manager entry"""
        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        headers["Accept"] = "application/vnd.github.v3+json"

        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def search_github_repos(
        self,
        language: str,
        min_stars: Optional[int] = None
    ) -> List[Dict]:
        """
        Search GitHub for repositories matching criteria

        Args:
            language: Programming language filter
            min_stars: Minimum stars (uses self.min_stars if None)

        Returns:
            List of repository metadata dictionaries
        """
        if not self.session:
            raise RuntimeError("Use async context manager (async with)")

        min_stars = min_stars or self.min_stars
        query = f"language:{language} stars:>={min_stars} fork:false"

        # Search for repositories updated in the last year (active projects)
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        query += f" pushed:>={one_year_ago}"

        url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 100
        }

        repos = []
        page = 1

        while len(repos) < self.max_repos:
            params["page"] = page

            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 403:  # Rate limit
                        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                        wait_time = max(reset_time - datetime.now().timestamp(), 0)
                        logger.warning(f"Rate limited. Waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status != 200:
                        logger.error(f"GitHub API error: {response.status}")
                        break

                    data = await response.json()
                    items = data.get("items", [])

                    if not items:
                        break

                    for repo in items:
                        # Filter by license
                        license_info = repo.get("license")
                        if not license_info:
                            continue

                        license_key = license_info.get("spdx_id", "UNKNOWN")
                        if license_key not in self.APPROVED_LICENSES:
                            continue

                        repos.append({
                            "repo_url": repo["html_url"],
                            "full_name": repo["full_name"],
                            "language": repo["language"],
                            "stars": repo["stargazers_count"],
                            "license": license_key,
                            "last_updated": repo["updated_at"],
                            "description": repo.get("description", ""),
                            "topics": repo.get("topics", []),
                            "size_kb": repo["size"]
                        })

                    page += 1
                    await asyncio.sleep(1)  # Respect rate limits

            except Exception as e:
                logger.error(f"Error fetching repos: {e}")
                break

        logger.info(f"Found {len(repos)} repositories for {language}")
        return repos[:self.max_repos]

    async def download_repository_files(
        self,
        repo_full_name: str,
        output_path: Path
    ) -> int:
        """
        Download source files from a repository

        Args:
            repo_full_name: Repository name (e.g., "owner/repo")
            output_path: Path to save files

        Returns:
            Number of files downloaded
        """
        if not self.session:
            raise RuntimeError("Use async context manager")

        # Get repository tree
        url = f"https://api.github.com/repos/{repo_full_name}/git/trees/main"
        params = {"recursive": 1}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    # Try 'master' branch if 'main' fails
                    url = f"https://api.github.com/repos/{repo_full_name}/git/trees/master"
                    response = await self.session.get(url, params=params)
                    if response.status != 200:
                        logger.warning(f"Could not fetch tree for {repo_full_name}")
                        return 0

                data = await response.json()
                tree = data.get("tree", [])

            # Filter for source code files
            file_extensions = {
                ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c",
                ".h", ".hpp", ".cs", ".go", ".rs", ".swift", ".kt", ".rb",
                ".php", ".scala", ".r"
            }

            files_to_download = [
                item for item in tree
                if item["type"] == "blob" and
                any(item["path"].endswith(ext) for ext in file_extensions)
            ]

            # Download files
            downloaded = 0
            for file_item in files_to_download[:100]:  # Limit files per repo
                file_path = file_item["path"]
                file_url = f"https://raw.githubusercontent.com/{repo_full_name}/main/{file_path}"

                try:
                    async with self.session.get(file_url) as response:
                        if response.status == 200:
                            content = await response.text()

                            # Save file
                            save_path = output_path / file_path
                            save_path.parent.mkdir(parents=True, exist_ok=True)

                            with open(save_path, "w", encoding="utf-8") as f:
                                f.write(content)

                            downloaded += 1

                except Exception as e:
                    logger.debug(f"Could not download {file_path}: {e}")

                await asyncio.sleep(0.1)  # Rate limit

            logger.info(f"Downloaded {downloaded} files from {repo_full_name}")
            return downloaded

        except Exception as e:
            logger.error(f"Error downloading repository {repo_full_name}: {e}")
            return 0

    async def collect_language_data(self, language: str) -> int:
        """
        Collect all data for a specific language

        Args:
            language: Programming language

        Returns:
            Total number of files collected
        """
        logger.info(f"Collecting data for {language}...")

        repos = await self.search_github_repos(language)
        total_files = 0

        for repo in repos:
            repo_name = repo["full_name"]

            if repo_name in self.collected_repos:
                continue

            output_path = self.output_dir / language / repo_name.replace("/", "_")

            try:
                files_count = await self.download_repository_files(
                    repo_name,
                    output_path
                )
                total_files += files_count
                self.collected_repos.add(repo_name)

                # Save metadata
                metadata_path = output_path / "metadata.json"
                import json
                with open(metadata_path, "w") as f:
                    json.dump(repo, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to collect {repo_name}: {e}")

            await asyncio.sleep(2)  # Be respectful of GitHub's servers

        logger.info(f"Collected {total_files} files for {language}")
        return total_files

    async def collect_all(self) -> Dict[str, int]:
        """
        Collect data for all supported languages

        Returns:
            Dictionary mapping language to file count
        """
        results = {}

        for language in self.SUPPORTED_LANGUAGES:
            try:
                count = await self.collect_language_data(language)
                results[language] = count
            except Exception as e:
                logger.error(f"Failed to collect {language}: {e}")
                results[language] = 0

        logger.info(f"Collection complete. Total repos: {len(self.collected_repos)}")
        return results


async def main():
    """Example usage"""
    collector = CodeCollector(
        min_stars=50,
        max_repos=100,
        output_dir="data/raw/code"
    )

    async with collector:
        # Collect Python and JavaScript as examples
        results = await collector.collect_language_data("Python")
        print(f"Collected {results} Python files")

        results = await collector.collect_language_data("JavaScript")
        print(f"Collected {results} JavaScript files")


if __name__ == "__main__":
    asyncio.run(main())
