const Comment = require("../models/comment");
const { Op } = require("sequelize");

const commentController = {
  // Add a comment
  async addComment(req, res) {
    try {
      const { content, postId, parentId } = req.body;
      const authorId = req.user.id;

      const comment = await Comment.create({
        content,
        postId,
        authorId,
        parentId: parentId || null,
      });

      res.status(201).json({
        message: "Comment added successfully",
        comment,
      });
    } catch (error) {
      console.error("Comment creation error:", error);
      res.status(500).json({ error: error.message });
    }
  },

  // List comments for a specific post
  async listComments(req, res) {
    try {
      const { post_id } = req.query;

      if (!post_id) {
        return res.status(400).json({ error: "post_id is required" });
      }

      const comments = await Comment.findAll({
        where: {
          postId: post_id,
          status: "active",
          parentId: null, // Only get top-level comments
        },
        order: [["createdAt", "DESC"]],
        attributes: {
          exclude: ["updatedAt"],
        },
      });

      res.json({
        comments,
        count: comments.length,
      });
    } catch (error) {
      console.error("Comment listing error:", error);
      res.status(500).json({ error: error.message });
    }
  },

  // Future method for nested comments
  async getNestedComments(req, res) {
    try {
      const { post_id } = req.query;
      const { parent_id } = req.query;

      const comments = await Comment.findAll({
        where: {
          postId: post_id,
          parentId: parent_id,
          status: "active",
        },
        order: [["createdAt", "ASC"]],
      });

      res.json(comments);
    } catch (error) {
      console.error("Nested comments fetch error:", error);
      res.status(500).json({ error: error.message });
    }
  },
};

module.exports = commentController;
