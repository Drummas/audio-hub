(function () {
  const root = document.documentElement;
  const stored = localStorage.getItem("theme");
  if (stored) {
    root.setAttribute("data-theme", stored);
  } else {
    root.setAttribute("data-theme", "dark");
  }

  window.toggleTheme = function () {
    const current = root.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
  };
})();
