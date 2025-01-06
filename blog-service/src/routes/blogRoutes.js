const express = require("express");
const router = express.Router();
const blogController = require("../controllers/blogController");
const authMiddleware = require("../middlewares/authMiddleware");
const { validateBlog } = require("../middlewares/validateMiddleware");

// Public routes
router.get("/blogs", blogController.listBlogs);
router.get("/blogs/:id", blogController.getBlog);

// Protected routes
router.post("/blogs", authMiddleware, validateBlog, blogController.createBlog);
router.put(
  "/blogs/:id",
  authMiddleware,
  validateBlog,
  blogController.updateBlog
);
router.delete("/blogs/:id", authMiddleware, blogController.deleteBlog);

module.exports = router;
