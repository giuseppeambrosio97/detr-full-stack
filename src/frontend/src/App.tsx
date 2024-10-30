import { Provider } from "react-redux";
import { HashRouter as Router, Route, Routes } from "react-router-dom";
import store from "@/redux/store";
import HomePage from "@/pages/home/HomePage";
import InferencePage from "@/pages/inference/InferencePage";
import Layout from "@/pages/Layout";
import ExportPage from "@/pages/export/ExportPage";

function App() {
  return (
    <Provider store={store}>
      <Router>
        <div className="w-screen h-screen">
          <Routes>
            {/* Layout Route */}
            <Route element={<Layout />}>
              <Route path="/home" element={<HomePage />} />
              <Route path="/inference" element={<InferencePage />} />
              <Route path="/export" element={<ExportPage />} />
              {/* Default Route */}
              <Route path="/" element={<HomePage />} />
            </Route>

          </Routes>
        </div>
      </Router>
    </Provider>
  );
}

export default App;
