import React from "react";
import DashboardLayout from "../../components/DashboardLayout";
import InLineScanner from "../../components/InLineScanner";

export default function InLineScannerPage() {
  return (
    <DashboardLayout>
      <div className="mb-6">
        <h2 className="text-xl font-bold text-slate-800 mb-1">In-Line Camera Scanner & Real-Time Inspection</h2>
        <p className="text-slate-500 text-sm">
          Simulated optical conveyor camera feed, batch image scanning, real-time anomaly detection, and severity assessment.
        </p>
      </div>

      <InLineScanner />
    </DashboardLayout>
  );
}
