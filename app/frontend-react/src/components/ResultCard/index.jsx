import React from "react";
import styles from "./ResultCard.module.css";

const ResultCard = ({ title, value }) => {
  const formattedValue = value
    .toFixed(0)
    .replace(/\B(?=(\d{3})+(?!\d))/g, " ");

  return (
    <div className={styles.resultCard}>
      <p className={styles.price}>
        Predicted price : <span className={styles.priceValue}>{formattedValue} €</span>
      </p>
    </div>
  );
};

export default ResultCard;
