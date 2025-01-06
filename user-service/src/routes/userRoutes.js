// src/routes/userRoutes.js
const express = require("express");
const router = express.Router();
const userController = require("../controllers/userController");
const authMiddleware = require("../middlewares/authMiddleware");
const { validateRegistration } = require("../middlewares/validateMiddleware");

// Public routes
router.post("/register", validateRegistration, userController.register);
router.post("/login", userController.login);

// Protected routes
router.get("/users/:id", authMiddleware, userController.getUser);
router.put("/users/:id", authMiddleware, userController.updateUser);
router.delete("/users/:id", authMiddleware, userController.deleteUser);

module.exports = router;
