import React from "react";
import { useTranslation } from "react-i18next";

interface ColumnSortProps {
  column: {
    field: string;
    label: string;
    type?: string;
  };
  currentSort: { field: string; direction: "asc" | "desc" } | null;
  onSortChange: (field: string) => void;
}

export const ColumnSort: React.FC<ColumnSortProps> = ({ column, currentSort, onSortChange }) => {
  const { t } = useTranslation("common");
  const isActive = currentSort?.field === column.field;
  const direction = currentSort?.direction || "asc";

  return (
    <button
      type="button"
      onClick={() => onSortChange(column.field)}
      style={{
        background: "transparent",
        border: "none",
        color: isActive ? "#fff" : "#94a3b8",
        cursor: "pointer",
        padding: "4px",
        borderRadius: "4px",
        fontSize: "12px",
        marginLeft: "4px",
        transition: "all 0.2s",
        display: "inline-flex",
        alignItems: "center",
      }}
      title={isActive && direction === "asc" ? t("sort.descending") : t("sort.ascending")}
    >
      {!isActive && "⇅"}
      {isActive && direction === "asc" && "↑"}
      {isActive && direction === "desc" && "↓"}
    </button>
  );
};