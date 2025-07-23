// server/index.js
const express = require("express");
const cors = require("cors");
const puppeteer = require("puppeteer");

const app = express();
app.use(cors());
app.use(express.json());

app.post("/add-to-cart", async (req, res) => {
  const { url } = req.body;

  if (!url || !url.startsWith("http")) {
    return res.status(400).json({ message: "Invalid URL" });
  }

  try {
    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();

    await page.goto(url, { waitUntil: "networkidle2" });

    // Attempt size selection (if available)
    try {
      await page.waitForSelector("select[name='dropdown_selected_size_name']", { timeout: 3000 });
      await page.select("select[name='dropdown_selected_size_name']", "M");
    } catch (e) {
      console.log("No size dropdown found");
    }

    // Attempt to click Add to Cart
    await page.waitForSelector("#add-to-cart-button", { timeout: 5000 });
    await page.click("#add-to-cart-button");

    await page.waitForTimeout(3000); // wait for page update

    const success = await page.evaluate(() => {
      return document.body.innerText.includes("Added to Cart") || document.body.innerText.includes("Proceed to checkout");
    });

    await browser.close();

    return res.json({
      message: success ? "✅ Product successfully added to cart!" : "❌ Could not confirm add to cart.",
    });
  } catch (err) {
    console.error("Error adding to cart:", err);
    return res.status(500).json({ message: "Automation failed." });
  }
});

app.listen(5000, () => console.log("🚀 Server running on http://localhost:5000"));
