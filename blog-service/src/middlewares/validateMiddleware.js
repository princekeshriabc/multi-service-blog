const validateBlog = (req, res, next) => {
  const { title, content } = req.body;

  if (!title || !content) {
    return res.status(400).json({
      error: "Title and content are required",
    });
  }

  if (title.length < 3) {
    return res.status(400).json({
      error: "Title must be at least 3 characters long",
    });
  }

  if (content.length < 10) {
    return res.status(400).json({
      error: "Content must be at least 10 characters long",
    });
  }

  next();
};

module.exports = {
  validateBlog,
};
