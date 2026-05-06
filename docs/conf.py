project = "Rehearsal"
author = "LAMDA, Nanjing University"
copyright = "2026, LAMDA, Nanjing University"
version = "0.1.5"
release = "0.1.5"

extensions = []
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "Rehearsal Documentation"
html_static_path = []
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
}
html_context = {
    "display_github": True,
    "github_user": "Rehearsal-LAMDA",
    "github_repo": "Rehearsal-Kit",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
