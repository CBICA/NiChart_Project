import React, { useState } from "react";

export const ModelSelectionMenu = props => {
  // State with list of all checked item
  const [checked, setChecked] = useState([]);
  const checkList = ["Alzheimer's Disease", "Brain Age"];

  // Add/Remove checked item from list
  const handleCheck = (event) => {
    var updatedList = [...checked];
    if (event.target.checked) {
      updatedList = [...checked, event.target.value];
    } else {
      updatedList.splice(checked.indexOf(event.target.value), 1);
    }
    setChecked(updatedList);
  };

  // Generate string of checked items
  const checkedItems = checked.length
    ? checked.reduce((total, item) => {
        return total + ", " + item;
      })
    : "";

  // Return classes based on whether item is checked
  var isChecked = (item) =>
    checked.includes(item) ? "checked-item" : "not-checked-item";

  return (
    <div className="model-selector">
      <div className="checkList">
        <div className="title">Available Models:</div>
        <div className="list-container">
          {checkList.map((item, index) => (
            <div key={index}>
              <input value={item} type="checkbox" onChange={handleCheck} />
              <span className={isChecked(item)}>{item}</span>
            </div>
            
          ))}
          <div>
            <p>Additional models will be available soon. Stay tuned!</p>
          </div>
        </div>
      </div>
      <br />
      <div hidden>
        <p>{`Items checked are: ${checkedItems}`}</p>
      </div>
    </div>
  );
}

export default ModelSelectionMenu;