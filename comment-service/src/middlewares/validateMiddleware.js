const validateComment = (req, res, next) => {
  const { content, postId } = req.body;

  if (!content || !postId) {
    return res.status(400).json({
      error: "Content and postId are required",
    });
  }

  if (content.length < 1 || content.length > 1000) {
    return res.status(400).json({
      error: "Content must be between 1 and 1000 characters",
    });
  }

  next();
};

module.exports = {
  validateComment,
};
