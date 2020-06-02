import cv2


def display_boxes(cfg, boxes, pred_labels, frame, palette, labels):
    for box, box_labels in zip(boxes.astype(int), pred_labels):
        cv2.rectangle(frame, tuple(box[:2]), tuple(
            box[2:]), (0, 255, 0), thickness=2)
        label_origin = box[:2]
        for label in box_labels:
            label_origin[-1] -= 5
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, .5, 2)
            cv2.rectangle(
                frame,
                (label_origin[0], label_origin[1] + 5),
                (label_origin[0] + label_width,
                 label_origin[1] - label_height - 5),
                palette[labels.index(label)], -1
            )
            cv2.putText(
                frame, label, tuple(label_origin),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )
            label_origin[-1] -= label_height + 5

    return frame


def display_text_label(cfg, frame, pred_labels):
    # Display predicted labels to frame.
    y_offset = 50
    cv2.putText(frame, 'Action:', (10, y_offset),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.65, color=(0, 235, 0), thickness=2)
    for pred_label in pred_labels:
        y_offset += 30
        cv2.putText(frame, '{}'.format(pred_label), (20, y_offset),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.65, color=(0, 235, 0), thickness=2)

    return frame
