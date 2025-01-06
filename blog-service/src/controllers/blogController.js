const Blog = require("../models/blog");

const blogController = {
  // Create new blog post
  async createBlog(req, res) {
    try {
      const { title, content, status } = req.body;
      const authorId = req.user.id;

      const blog = await Blog.create({
        title,
        content,
        authorId,
        status: status || "published",
      });

      res.status(201).json({
        message: "Blog post created successfully",
        blog,
      });
    } catch (error) {
      console.error("Blog creation error:", error);
      res.status(500).json({ error: error.message });
    }
  },

  // List all blog posts with pagination
  async listBlogs(req, res) {
    try {
      const { page = 1, limit = 10 } = req.query;

      const options = {
        page: parseInt(page),
        paginate: parseInt(limit),
        order: [["createdAt", "DESC"]],
        where: {
          status: "published",
        },
      };

      const { docs, pages, total } = await Blog.paginate(options);

      res.json({
        blogs: docs,
        currentPage: page,
        totalPages: pages,
        totalItems: total,
      });
    } catch (error) {
      console.error("Blog listing error:", error);
      res.status(500).json({ error: error.message });
    }
  },

  // Get specific blog post
  async getBlog(req, res) {
    try {
      const blog = await Blog.findByPk(req.params.id);

      if (!blog) {
        return res.status(404).json({ error: "Blog post not found" });
      }

      res.json(blog);
    } catch (error) {
      console.error("Blog fetch error:", error);
      res.status(500).json({ error: error.message });
    }
  },

  // Update blog post
  async updateBlog(req, res) {
    try {
      const { title, content, status } = req.body;
      const blog = await Blog.findByPk(req.params.id);

      if (!blog) {
        return res.status(404).json({ error: "Blog post not found" });
      }

      if (blog.authorId !== req.user.id) {
        return res
          .status(403)
          .json({ error: "Unauthorized to edit this blog post" });
      }

      await blog.update({
        title: title || blog.title,
        content: content || blog.content,
        status: status || blog.status,
      });

      res.json({
        message: "Blog post updated successfully",
        blog,
      });
    } catch (error) {
      console.error("Blog update error:", error);
      res.status(500).json({ error: error.message });
    }
  },

  // Delete blog post
  async deleteBlog(req, res) {
    try {
      const blog = await Blog.findByPk(req.params.id);

      if (!blog) {
        return res.status(404).json({ error: "Blog post not found" });
      }

      if (blog.authorId !== req.user.id) {
        return res
          .status(403)
          .json({ error: "Unauthorized to delete this blog post" });
      }

      await blog.destroy();
      res.json({ message: "Blog post deleted successfully" });
    } catch (error) {
      console.error("Blog deletion error:", error);
      res.status(500).json({ error: error.message });
    }
  },
};

module.exports = blogController;
