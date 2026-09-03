import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { useAuth } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";

import Login from "./pages/Login";
import Register from "./pages/Register";
import Profile from "./pages/Profile";

import QEDashboard from "./pages/QualityEngineer/Dashboard";
import UploadImage from "./pages/QualityEngineer/UploadImage";
import InspectionHistory from "./pages/QualityEngineer/InspectionHistory";
import QualityReports from "./pages/QualityEngineer/QualityReports";
import InLineScannerPage from "./pages/QualityEngineer/InLineScannerPage";

import FSDashboard from "./pages/FactorySupervisor/Dashboard";
import InspectionReports from "./pages/FactorySupervisor/InspectionReports";
import DefectTrends from "./pages/FactorySupervisor/DefectTrends";
import QualityAnalytics from "./pages/FactorySupervisor/QualityAnalytics";
import ProductionMonitoring from "./pages/FactorySupervisor/ProductionMonitoring";
import UserManagement from "./pages/FactorySupervisor/UserManagement";

// Renders the correct dashboard depending on the logged-in user's role
function RoleDashboard() {
  const { user } = useAuth();
  return user?.role === "factory_supervisor" ? <FSDashboard /> : <QEDashboard />;
}

export default function App() {
  const { user } = useAuth();

  return (
    <Routes>
      <Route path="/login" element={user ? <Navigate to="/dashboard" /> : <Login />} />
      <Route path="/register" element={user ? <Navigate to="/dashboard" /> : <Register />} />

      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <RoleDashboard />
          </ProtectedRoute>
        }
      />

      <Route
        path="/inline-scanner"
        element={
          <ProtectedRoute>
            <InLineScannerPage />
          </ProtectedRoute>
        }
      />

      {/* Quality Engineer only */}
      <Route
        path="/upload"
        element={
          <ProtectedRoute allowedRoles={["quality_engineer"]}>
            <UploadImage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/inspections"
        element={
          <ProtectedRoute allowedRoles={["quality_engineer"]}>
            <InspectionHistory />
          </ProtectedRoute>
        }
      />
      <Route
        path="/reports"
        element={
          <ProtectedRoute allowedRoles={["quality_engineer"]}>
            <QualityReports />
          </ProtectedRoute>
        }
      />

      {/* Factory Supervisor only */}
      <Route
        path="/inspection-reports"
        element={
          <ProtectedRoute allowedRoles={["factory_supervisor"]}>
            <InspectionReports />
          </ProtectedRoute>
        }
      />
      <Route
        path="/defect-trends"
        element={
          <ProtectedRoute allowedRoles={["factory_supervisor"]}>
            <DefectTrends />
          </ProtectedRoute>
        }
      />
      <Route
        path="/quality-analytics"
        element={
          <ProtectedRoute allowedRoles={["factory_supervisor"]}>
            <QualityAnalytics />
          </ProtectedRoute>
        }
      />
      <Route
        path="/production-monitoring"
        element={
          <ProtectedRoute allowedRoles={["factory_supervisor"]}>
            <ProductionMonitoring />
          </ProtectedRoute>
        }
      />
      <Route
        path="/user-management"
        element={
          <ProtectedRoute allowedRoles={["factory_supervisor"]}>
            <UserManagement />
          </ProtectedRoute>
        }
      />

      {/* Shared */}
      <Route
        path="/profile"
        element={
          <ProtectedRoute>
            <Profile />
          </ProtectedRoute>
        }
      />

      <Route path="*" element={<Navigate to={user ? "/dashboard" : "/login"} />} />
    </Routes>
  );
}
