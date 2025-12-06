// scripts.js
// Basic interactivity for Weather Aggregator frontend

document.addEventListener("DOMContentLoaded", function() {
    console.log("Weather Aggregator frontend loaded ✅");

    // Example: flash message auto-hide
    const flashes = document.querySelectorAll(".flash-message");
    flashes.forEach(msg => {
        setTimeout(() => {
            msg.style.display = "none";
        }, 4000);
    });

    // Example: toggle forecast section
    const toggleBtn = document.getElementById("toggle-forecast");
    if (toggleBtn) {
        toggleBtn.addEventListener("click", () => {
            const section = document.querySelector(".forecast-section");
            if (section.style.display === "none") {
                section.style.display = "block";
                toggleBtn.innerText = "Hide Forecasts";
            } else {
                section.style.display = "none";
                toggleBtn.innerText = "Show Forecasts";
            }
        });
    }
});
