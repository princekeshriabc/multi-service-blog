const express = require("express");
const router = express.Router();
const commentController = require("../controllers/commentController");
const authMiddleware = require("../middlewares/authMiddleware");
const { validateComment } = require("../middlewares/validateMiddleware");

// Public routes
router.get("/comments", commentController.listComments);

// Protected routes
router.post(
  "/comments",
  authMiddleware,
  validateComment,
  commentController.addComment
);

// Future nested comments route
// router.get('/comments/nested', commentController.getNestedComments);

module.exports = router;
